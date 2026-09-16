from collections import defaultdict

import duckdb
import torch
from torch.utils.data import Dataset


EVENT_COLUMNS = ("patient_id", "index", *(f"token_{i}" for i in range(9)))

EVENT_TYPES = {
    "encounter": "encounters", "condition": "conditions", "medication": "medications",
    "procedure": "procedures", "observation": "observations", "immunization": "immunizations",
    "allergy": "allergies", "careplan": "careplans", "imaging_study": "imaging_studies",
}
EVENT_TO_INDEX = {"patients": 0, **{table: i + 1 for i, table in enumerate(EVENT_TYPES.values())}}
INDEX_TO_EVENT = {v: k for k, v in EVENT_TO_INDEX.items()}

TIME_SCALE = (60*60*24*365) * 50  # 50 years in seconds

class PatientEventsDataset(Dataset):
    def __init__(self, path, context_length, max_poison_fraction=0.2, threads = 1):
        self.path = path
        self.context_length = context_length
        self.threads = threads
        self.max_poison_fraction = max_poison_fraction

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
        if not indices:
            return []

        batch_patient_data, batch_events_by_patient = self.get_data_from_database(indices)
        new_indices, labels = self.construct_negative_pairs(indices)


        return [self._get_sample(pos, neg, batch_patient_data, batch_events_by_patient)
                for pos, neg in new_indices]

    def get_data_from_database(self, indices):
        batch_patient_data = [self.index[index] for index in indices]
        batch_events_by_patient = self._get_events_data_by_indices(batch_patient_data)
        return batch_patient_data, batch_events_by_patient

    def construct_negative_pairs(self, indices):
        # indices = 3k, last 1/3 is negative samples to be paired with the first 1/3 of indices
        positive = indices[:len(indices) // 3]
        negative = indices[len(indices) // 3:2 * len(indices) // 3]
        negative_poison = indices[2 * len(indices) // 3:]
        new_indices = [(pos, pos) for pos in positive]
        new_indices += [(pos, neg) for pos, neg in zip(positive, negative_poison)]
        labels = torch.tensor([1] * len(positive) + [0] * len(negative_poison), dtype=torch.float32)
        return new_indices, labels


    def construct_negative_sample(self, p_tokens, p_padding_mask, n_tokens, n_padding_mask):
        positive_token_length = (1 - p_padding_mask.float()).sum().item()
        select_token_length = int(positive_token_length * self.max_poison_fraction)


    def _get_sample(self, pos, neg, batch_patient_data, batch_events_by_patient): 
        tokens, padding_mask = self._get_patient_tokens(batch_patient_data, batch_events_by_patient, pos)

        if neg != pos:
            neg_tokens, neg_padding_mask = self._get_patient_tokens(batch_patient_data, batch_events_by_patient, neg)

        return tokens, padding_mask



    def _get_patient_tokens(self, batch_patient_data, batch_events_by_patient, index):
        patient = batch_patient_data[index]
        events = batch_events_by_patient[patient["patient_id"]]
        patient_tokens = self._patient_to_tensor(patient)
        event_tokens = self._events_to_tensor(events)
        tokens = torch.cat((patient_tokens, event_tokens), dim=1)
        tokens, padding_mask = self._enforce_context_length(tokens)
        return tokens, padding_mask

    def _enforce_context_length(self, tokens, shuffle=True):
        num_tokens, context_length = tokens.shape
        padding_mask = torch.zeros(self.context_length, dtype=torch.bool)
        if context_length > self.context_length:
            #keep the first 6 tokens (patient + first 5 events) and randomly sample the remaining tokens to fit within the context length
            if shuffle:
                num_tokens_to_keep = self.context_length - 6
                if num_tokens_to_keep > 0:
                    remaining_indices = torch.randperm(num_tokens - 6)[:num_tokens_to_keep] + 6
                    selected_indices = torch.cat((torch.arange(6), remaining_indices))
                    #shuffle the selected indices to randomize the order of the tokens
                    selected_indices = selected_indices[torch.randperm(len(selected_indices))]
                    tokens = tokens[:, selected_indices]
            else:
                #select a random time window of tokens to fit within the context length keeping the first 6 tokens (patient + first 5 events)
                start_index = torch.randint(6, num_tokens - self.context_length + 1, (1,)).item()
                end_index = start_index + self.context_length
                tokens = tokens[:, torch.cat((torch.arange(6), torch.arange(start_index, end_index)))]
        else:
            #pad the tokens to fit within the context length
            padding = self.context_length - context_length
            tokens = torch.cat((tokens, torch.zeros((num_tokens, padding))), dim=1)
            padding_mask[-padding:] = True
        return tokens, padding_mask

    @staticmethod
    def _events_to_tensor(events):
        if not events:
            return torch.empty((9, 0), dtype=torch.float32)
        event_tokens = torch.tensor([[event[f"token_{i}"] for event in events] for i in range(9)], dtype=torch.float32)
        event_tokens[3 : 5] /= TIME_SCALE  # Convert seconds to scaled time for start_time
        return event_tokens
    

    @staticmethod
    def _patient_to_tensor(patient):
        patient_tokens = torch.zeros((9, 1), dtype=torch.float32)
        for i in range(9):
            patient_tokens[i, 0] = patient[f"patient_token_{i}"]
        patient_tokens[3 : 5] /= TIME_SCALE  # Convert seconds to scaled time for start_time and end_time
        return patient_tokens




    def _get_events_data_by_indices(self, patient_data):
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
  