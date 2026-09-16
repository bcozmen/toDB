import duckdb
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


EVENT_COLUMNS = (
    "patient_id", "encounter_id", "start_time", "stop_time",
    "event_type", "code", "value", "units",
)
EVENT_DISCRETE_COLUMNS = (
    "code"
)
EVENT_CONTINUOUS_COLUMNS = (
    "value"
)
PATIENT_DISCRETE_COLUMNS = (
   "gender","race", "ethnicity", "birthplace", "city"
)
PATIENT_CONTINUOUS_COLUMNS = (
    "income",
)

EVENT_TO_INDEX = {
    "patients": 0,
    "encounters": 1,
    "conditions": 2,
    "medications": 3,
    "procedures": 4,
    "observations": 5,
    "immunizations": 6,
    "allergies": 7,
    "careplans": 8,
    "imaging_studies": 9,
}

INDEX_TO_EVENT = {v: k for k, v in EVENT_TO_INDEX.items()}

class PatientEventsDataset(Dataset):
    def __init__(self, path, context_length, threads = 1):
        self.path = path
        self.context_length = context_length
        self.threads = threads

        self.connection = None
        self.executor = None
        
        
        self.init_mode("train")  # Default mode is 'train'
        self.vocab = HealthcareVocab(f"{self.path}/vocab.pt")

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

    def _get_executor(self):
        if self.executor is None:
            self.executor = ThreadPoolExecutor(max_workers=self.threads)
        return self.executor

    
        
    def __len__(self):
        return len(self.index)
    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        """Load a batch whose patients are in the same Parquet part."""
        if not indices:
            return []

        batch_patient_data = [self.index[index] for index in indices]
        batch_events_by_patient = self.get_events_data_by_indices(batch_patient_data)

        def process_patient(row):
            events = batch_events_by_patient.get(row["patient_id"], [])
            return self._get_patient_from_events(row, events)

        executor = self._get_executor()
        return list(executor.map(process_patient, batch_patient_data))
    def __del__(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
    def _get_patient_from_events(self, patient_data, events_data):
        p_tokens = self.get_patient_tokens(patient_data)
        e_tokens = self.get_events(events_data)
        tokens = torch.cat((p_tokens, e_tokens), dim=1)

        mask = torch.ones(tokens.size(1), dtype=torch.float32)
        if tokens.size(1) < self.context_length:
            pad_len = self.context_length - tokens.size(1)
            tokens = torch.cat(
                [tokens, torch.zeros(tokens.size(0), pad_len, dtype=tokens.dtype)],
                dim=1,
            )
            mask = torch.cat([mask, torch.zeros(pad_len)], dim=0)
        return tokens, mask

    def get_events_data_by_indices(self, patient_data):
        """Read the requested patients from one Parquet part in one query."""
        file_path = f"{self.path}/{self.mode}/{patient_data[0]['event_file']}"
        predicates = []
        params = []
        max_events = self.context_length - 8

        for row in patient_data:
            start_index = max(0, row["n_events"] - max_events)
            predicates.append('(patient_id = ? AND "index" >= ? AND "index" < ?)')
            params.extend([row["patient_id"], start_index, row["n_events"]])

        query = f"""
            SELECT patient_id, "index", {', '.join(EVENT_COLUMNS[1:])}
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
        
class PatientEventsDataset(Dataset):
    def __init__(self, path, context_length, threads = 1):
        self.path = path
        self.context_length = context_length
        self.threads = threads
        self.connection = None
        self.init_mode("train")  # Default mode is 'train'
        self.vocab = HealthcareVocab(f"{self.path}/vocab.pt")

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

    def _get_patient(self, index):
        patient_data = self.index[index]
        p_tokens = self.get_patient_tokens(patient_data)
        e_tokens = self.get_events(patient_data)
        tokens = torch.cat((p_tokens, e_tokens), dim=1)

        mask = torch.ones(tokens.size(1), dtype=torch.float32)
        if tokens.size(1) < self.context_length:
            pad_len = self.context_length - tokens.size(1)
            tokens = torch.cat(
                [tokens, torch.zeros(tokens.size(0), pad_len, dtype=tokens.dtype, device=tokens.device)],
                dim=1,
            )
            mask = torch.cat(
                [mask, torch.zeros(pad_len, dtype=mask.dtype, device=mask.device)],
                dim=0,
            )
        return tokens, mask

    def __getitems__(self, indices):
        """Load a batch whose patients are in the same Parquet part."""
        if not indices:
            return []

        patient_data = [self.index[index] for index in indices]
        event_file = patient_data[0]["event_file"]
        if any(row["event_file"] != event_file for row in patient_data):
            raise ValueError("All indices in a batch must use the same event file")

        events_by_patient = self.get_events_data_by_indices(patient_data)
        return [
            self._get_patient_from_events(row, events_by_patient.get(row["patient_id"], []))
            for row in patient_data
        ]

    def _get_patient_from_events(self, patient_data, events_data):
        p_tokens = self.get_patient_tokens(patient_data)
        e_tokens = self.get_events(events_data)
        tokens = torch.cat((p_tokens, e_tokens), dim=1)

        mask = torch.ones(tokens.size(1), dtype=torch.float32)
        if tokens.size(1) < self.context_length:
            pad_len = self.context_length - tokens.size(1)
            tokens = torch.cat(
                [tokens, torch.zeros(tokens.size(0), pad_len, dtype=tokens.dtype)],
                dim=1,
            )
            mask = torch.cat([mask, torch.zeros(pad_len)], dim=0)
        return tokens, mask

    def get_events_data_by_indices(self, patient_data):
        """Read the requested patients from one Parquet part in one query."""
        file_path = f"{self.path}/{self.mode}/{patient_data[0]['event_file']}"
        predicates = []
        params = []
        max_events = self.context_length - 8

        for row in patient_data:
            start_index = max(0, row["n_events"] - max_events)
            predicates.append('(patient_id = ? AND "index" >= ? AND "index" < ?)')
            params.extend([row["patient_id"], start_index, row["n_events"]])

        query = f"""
            SELECT patient_id, "index", {', '.join(EVENT_COLUMNS[1:])}
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
    def _sample_negative_patient(self, exclude_idx):
        index = torch.randint(0, len(self.index) - 1, (1,)).item()
        if index == exclude_idx:
            index += 1
        return index
    def _sample_set(self, max_length):
        sampled_set = torch.randint(0, 3, (self.context_length - 7,))  + 1
        sampled_set[max_length:] = 0  # Zero out tokens beyond the max_length
        return sampled_set
    def __getitem__(self, idx):

        positive_patient, positive_mask = self._get_patient(idx)

        if torch.rand(1).item() < 0.5 and False:
            negative_idx = self._sample_negative_patient(idx)
            negative_patient, negative_mask = self._get_patient(negative_idx)
        
        return positive_patient, positive_mask


    def _encode(self, value, mode):
        if mode == "time":
            if str(value) == "NaT" or value is None:
                return torch.inf
            return value.timestamp() / 1e9  # Convert to seconds
        elif mode == "continuous":
            if value is None:
                return torch.nan
            return float(value)
        elif mode == "categorical":
            if value is None:
                return self.vocab.encode(None)
            return self.vocab.encode(str(value))

    def _singular_to_plural(self, singular):
        if singular == 'allergy':
            return 'allergies'
        elif singular == 'imaging_study':
            return 'imaging_studies'
        return singular + 's'

    def get_patient_tokens(self, patient_data):
        total_columns = len(PATIENT_DISCRETE_COLUMNS) + len(PATIENT_CONTINUOUS_COLUMNS)
        tokens = torch.zeros((8, total_columns), dtype=torch.float32)

        
        tokens[0] = float('nan')  # Placeholder for discrete token values
        tokens[0, :len(PATIENT_DISCRETE_COLUMNS)] = torch.tensor([self._encode(f"patients.{col}.{patient_data[col]}", "categorical") for col in PATIENT_DISCRETE_COLUMNS])
        
        tokens[1] = float('inf')  # Placeholder for continious token values
        tokens[1, len(PATIENT_DISCRETE_COLUMNS):] = torch.tensor([self._encode(patient_data[col], "continuous") for col in PATIENT_CONTINUOUS_COLUMNS])

        # Continuous mask: 1 for continuous columns, 0 for discrete columns
        tokens[2] = 0
        tokens[2, len(PATIENT_DISCRETE_COLUMNS):] = 1

        # Time encoding
        tokens[3] = self._encode(patient_data["patient_start_time"], "time")
        tokens[4] = self._encode(patient_data["patient_stop_time"], "time")

        # Table and column encoding
        tokens[5] = self._encode('patients', "categorical")
        tokens[6] = torch.tensor([self._encode(f"patients.{col}", "categorical") for col in PATIENT_DISCRETE_COLUMNS + PATIENT_CONTINUOUS_COLUMNS])
        
        # Info
        tokens[7] = EVENT_TO_INDEX['patients']  # Index for patients table
        return tokens

    
    
    def get_events(self, patient_data):
        events_data = self.get_events_data(patient_data)

        tokens = torch.zeros((8,len(events_data)), dtype=torch.float32)
        tokens[0] = float('nan')  # Placeholder for token values
        tokens[1] = float('inf')  # Placeholder for continuous token values
        for i, event in enumerate(events_data):
            table_name = self._singular_to_plural(event['event_type'])
            column_name = event['event_type']
            tokens[0, i] = self._encode(f"{table_name}.code.{event['code']}", "categorical")
            tokens[1, i] = self._encode(event['value'], "continuous")
            if column_name == "observation":
                tokens[2, i] = 1
            
            
            
            tokens[3, i] = self._encode(event['start_time'], "time")
            tokens[4, i] = self._encode(event['stop_time'], "time")

            tokens[5, i] = self._encode(f"{table_name}", "categorical")
            tokens[7, i] = EVENT_TO_INDEX[table_name]
            
            #tokens[6, i] = self._encode(f"{table_name}.'code", "categorical")

        
        return tokens
    
    def get_events_data(self, patient_data):
        n_events = patient_data['n_events']
        if n_events > self.context_length - 7:
            start_index = n_events - (self.context_length - 7)
        else:
            start_index = 0
        end_index = start_index + min(n_events, self.context_length - 7)
        return self.get_events_data_by_index(patient_data, start_index, end_index)
    def get_events_data_by_time(self, patient_data):
        file_path = f"{self.path}/{self.mode}/{patient_data['event_file']}"
        query = f"""
            SELECT {', '.join(EVENT_COLUMNS)}
            FROM read_parquet('{file_path}')
                        WHERE patient_id = ?
                            AND start_time >= ?
                            AND start_time < ?
        """
        res = self._get_connection().execute(query, [patient_data['patient_id'], patient_data['first_event_time'], patient_data['last_event_time']])
        cols = [col[0] for col in res.description]
        events = [dict(zip(cols, row)) for row in res.fetchall()]

        return events

    def get_events_data_by_index(self, patient_data, start_index, end_index):
        """Load a patient's events whose sorted indices are in [start_index, end_index)."""
        file_path = f"{self.path}/{self.mode}/{patient_data['event_file']}"
        query = f"""
            SELECT {', '.join(EVENT_COLUMNS)}
            FROM read_parquet('{file_path}')
            WHERE patient_id = ?
                AND "index" >= ?
                AND "index" < ?
            ORDER BY "index"
        """
        res = self._get_connection().execute(
            query,
            [patient_data['patient_id'], start_index, end_index],
        )
        cols = [col[0] for col in res.description]
        return [dict(zip(cols, row)) for row in res.fetchall()]
