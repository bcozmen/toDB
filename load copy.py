from collections import defaultdict

import duckdb
import torch
from torch.utils.data import Dataset


EVENT_COLUMNS = ("patient_id", "index", *(f"token_{i}" for i in range(8)))

EVENT_TYPES = {
    "encounter": "encounters", "condition": "conditions", "medication": "medications",
    "procedure": "procedures", "observation": "observations", "immunization": "immunizations",
    "allergy": "allergies", "careplan": "careplans", "imaging_study": "imaging_studies",
}
EVENT_TO_INDEX = {"patients": 0, **{table: i + 1 for i, table in enumerate(EVENT_TYPES.values())}}
INDEX_TO_EVENT = {v: k for k, v in EVENT_TO_INDEX.items()}

class PatientEventsDataset(Dataset):
    def __init__(self, path, context_length, threads = 1):
        self.path = path
        self.context_length = context_length
        self.threads = threads

        self.connection = None
        self.init_mode("train")

    def init_mode(self, mode):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

        self.mode = mode
        with duckdb.connect(":memory:") as con:
            self.index = con.execute(
                f"SELECT * FROM read_parquet('{self.path}/{mode}/patient_index.parquet')"
            ).fetch_df().to_dict("records")
        self.patients = {row["patient_id"]: row for row in self.index}

    def _get_connection(self):
        if self.connection is None:
            self.connection = duckdb.connect(":memory:")
            self.connection.execute(f"SET threads={self.threads}")
        return self.connection


    
        
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        """Load a batch whose patients are in the same pre-tokenized part."""
        if not indices:
            return []

        batch_patient_data = [self.index[index] for index in indices]
        batch_events_by_patient = self.get_events_data_by_indices(batch_patient_data)

        new_indices = self._construct_negative_pairs(indices)        

        return [self._get_sample(ix, target, batch_patient_data, batch_events_by_patient)
                for ix, target in new_indices]

    def _construct_negative_pairs(self, indices):
        shuffled_indices = torch.randperm(len(indices))
        indices = [indices[i] for i in shuffled_indices]
        pairs = [(indices[i], indices[i + 1]) for i in range(0, len(indices) - 1, 2)]

        new_indices = [[index, index] for index in indices]
        for i, j in pairs:
            if new_indices[i][0] == i:
                new_indices[i][1] = j
        return new_indices

    def _get_relational_mask(self, tokens):
        #6 patient + 9 event tokens of 3 types
        mask_template = torch.randint(0, 3, (6 + 9,))
        mask = torch.zeros_like(tokens)
        mask[0:6]
        return mask

    def _get_temporal_mask(self, tokens):
        columns = tokens[7]
        encounter_indices = columns == EVENT_TO_INDEX["encounters"]

        mask = torch.randint(0, 3, (len(encounter_indices),))
        return mask

    def fit_to_context_length(self, tokens):
        padding_mask = torch.zeros(tokens.size(1), dtype=torch.float32)

        if tokens.size(1) > self.context_length:
            columns = tokens[7]
            encounter_indices = torch.where(columns == EVENT_TO_INDEX["encounters"])[0]
            #
        elif tokens.size(1) < self.context_length:
            padding = self.context_length - tokens.size(1)
            tokens = torch.cat((tokens, torch.zeros((8, padding))), dim=1)
            padding_mask = torch.cat((padding_mask, torch.ones(padding, dtype=torch.float32)))
        return tokens, padding_mask

    def _get_sample(self, ix, target, batch_patient_data, batch_events_by_patient):
        patient = batch_patient_data[ix]
        events = batch_events_by_patient[patient["patient_id"]]
        patient_tokens = self._get_patient_tokens(patient, events)

        negative = ix != target
        temporal = torch.rand(1).item() < 0.5

        if negative:
            target_patient = batch_patient_data[target]
            target_events = batch_events_by_patient[target_patient["patient_id"]]
            target_tokens = self._get_patient_tokens(target_patient, target_events)

        patient_tokens = self._patient_to_tensor(patient)
        event_tokens = self._events_to_tensor(events)
        
        tokens = torch.cat((patient_tokens, event_tokens), dim=1)
        mask = torch.ones(tokens.size(1), dtype=torch.float32)
        if tokens.size(1) < self.context_length:
            padding = self.context_length - tokens.size(1)
            tokens = torch.cat((tokens, torch.zeros((8, padding))), dim=1)
            mask = torch.cat((mask, torch.zeros(padding)))
        return tokens, mask

    def _get_patient_tokens(self, patient, events):
        patient_tokens = self._patient_to_tensor(patient)
        event_tokens = self._events_to_tensor(events)
        
        tokens = torch.cat((patient_tokens, event_tokens), dim=1)
        #tokens = tokens[:, :self.context_length]
        #mask = torch.ones(tokens.size(1), dtype=torch.float32)
        #if tokens.size(1) < self.context_length:
        #    padding = self.context_length - tokens.size(1)
        #    tokens = torch.cat((tokens, torch.zeros((8, padding))), dim=1)
        #    mask = torch.cat((mask, torch.zeros(padding)))
        return tokens

    @staticmethod
    def _events_to_tensor(events):
        if not events:
            return torch.empty((8, 0), dtype=torch.float32)
        event_tokens = torch.tensor([[event[f"token_{i}"] for event in events] for i in range(8)], dtype=torch.float32)
        event_tokens[3] /= 1e9  # Convert nanoseconds to seconds for start_time
        event_tokens[4] /= 1e9  # Convert nanoseconds to seconds for
        return event_tokens
    

    @staticmethod
    def _patient_to_tensor(patient):
        patient_tokens = torch.zeros((8, 1), dtype=torch.float32)
        for i in range(8):
            patient_tokens[i, 0] = patient[f"patient_token_{i}"]
        patient_tokens[3] /= 1e9  # Convert nanoseconds to seconds for start_time
        patient_tokens[4] /= 1e9  # Convert nanoseconds to seconds for
        return patient_tokens

    def get_events_data_by_indices(self, patient_data):
        """Read the requested patients from one Parquet part in one query."""
        file_path = f"{self.path}/{self.mode}/{patient_data[0]['event_file']}"
        predicates = []
        params = []
        max_events = self.context_length - 6

        for row in patient_data:
            start_index = max(0, row["n_events"] - max_events)
            predicates.append('(patient_id = ? AND "index" >= ? AND "index" < ?)')
            params.extend([row["patient_id"], start_index, row["n_events"]])

        query = f"""
            SELECT {', '.join(EVENT_COLUMNS)}
            FROM read_parquet('{file_path}')
            WHERE {' OR '.join(predicates)}
            ORDER BY patient_id, "index"
        """
        res = self._get_connection().execute(query, params)
        cols = [col[0] for col in res.description]
        events_by_patient = defaultdict(list)
        for values in res.fetchall():
            event = dict(zip(cols, values))
            events_by_patient[event["patient_id"]].append(event)
        return events_by_patient
  