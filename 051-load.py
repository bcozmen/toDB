from collections import defaultdict
import duckdb
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

EVENT_COLUMNS = ("patient_id", "index", *(f"token_{i}" for i in range(9)))
NUM_CHANNELS = 9
NUM_PATIENT_TOKENS = 7
TIME_SCALE = (60 * 60 * 24 * 365) * 50
MAX_DATE = 1790812800 / TIME_SCALE


class PatientRepository:
    """Handles connection management and queries for patient/event parquet files."""
    
    def __init__(self, base_path: str, threads: int = 1, lazy_index: bool = False):
        self.base_path = base_path
        self.threads = threads
        self.lazy_index = lazy_index
        self.connection = None
        self._patient_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["connection"] = None  # Don't pickle the connection
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.connection = None  # Ensure connection is None after unpickling

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.clear_cache()

    def clear_cache(self):
        self._patient_cache.clear()

    def _get_connection(self):
        if self.connection is None:
            self.connection = duckdb.connect(":memory:")
            self.connection.execute(f"SET threads={self.threads}")
        return self.connection

    def get_patient_count(self, mode: str) -> int:
        """Fast O(1) row count check without loading data into memory."""
        path = f"{self.base_path}/{mode}/patient_index.parquet"
        with duckdb.connect(":memory:") as con:
            return con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]

    def load_patient_index(self, mode: str) -> list[dict]:
        """Eagerly loads full index into memory."""
        path = f"{self.base_path}/{mode}/patient_index.parquet"
        with duckdb.connect(":memory:") as con:
            return con.execute(f"SELECT * FROM read_parquet('{path}')").fetch_df().to_dict("records")

    def fetch_event_file_groups(self, mode: str) -> list[list[int]]:
        """Fast projection scan: reads ONLY the event_file column to build index groups in ms."""
        path = f"{self.base_path}/{mode}/patient_index.parquet"
        query = f"""
            SELECT event_file, (ROW_NUMBER() OVER () - 1) AS row_id
            FROM read_parquet('{path}')
        """
        res = self._get_connection().execute(query)
        
        groups = defaultdict(list)
        for event_file, row_id in res.fetchall():
            groups[event_file].append(row_id)
        return list(groups.values())

    def fetch_patients_by_indices(self, mode: str, indices: list[int]) -> list[dict]:
        """Fetches patient index rows on demand, caching loaded records."""
        missing_indices = [i for i in indices if i not in self._patient_cache]
        
        if missing_indices:
            path = f"{self.base_path}/{mode}/patient_index.parquet"
            placeholders = ", ".join(["?"] * len(missing_indices))
            
            # Query specific row positions from Parquet on-demand
            query = f"""
                WITH indexed AS (
                    SELECT *, (ROW_NUMBER() OVER () - 1) AS row_id
                    FROM read_parquet('{path}')
                )
                SELECT * FROM indexed
                WHERE row_id IN ({placeholders})
            """
            res = self._get_connection().execute(query, missing_indices)
            cols = [col[0] for col in res.description]
            
            for values in res.fetchall():
                row = dict(zip(cols, values))
                row_id = row.pop("row_id")
                self._patient_cache[row_id] = row

        return [self._patient_cache[i] for i in indices]


    def fetch_events_by_patients_vectorized(self, mode: str, patient_data: list[dict]) -> dict[int, torch.Tensor]:
        """Fetch and group patient events using DuckDB/NumPy vectorized operations."""
        if not patient_data:
            return {}

        file_path = f"{self.base_path}/{mode}/{patient_data[0]['event_file']}"
        patient_ids = [row["patient_id"] for row in patient_data]
        placeholders = ", ".join(["?"] * len(patient_ids))

        query = f"""
            SELECT patient_id, {', '.join(f'token_{i}' for i in range(NUM_CHANNELS))}
            FROM read_parquet('{file_path}')
            WHERE patient_id IN ({placeholders})
            ORDER BY patient_id, "index"
        """
        np_dict = self._get_connection().execute(query, patient_ids).fetchnumpy()

        if not np_dict or len(np_dict["patient_id"]) == 0:
            return {}

        p_ids = np.asarray(np_dict["patient_id"])
        token_columns = np.column_stack(
            [np.asarray(np_dict[f"token_{i}"]) for i in range(NUM_CHANNELS)]
        )

        events_by_patient = {
            pid.item() if isinstance(pid, np.generic) else pid:
            torch.from_numpy(token_columns[p_ids == pid].T).float()
            for pid in np.unique(p_ids)
        }
        return events_by_patient


class EventTensorizer:
    def __init__(self, num_channels, time_scale):
        self.num_channels = num_channels
        self.time_scale = time_scale
    def patient_to_tensor(self, patient):
        patient_tokens = torch.tensor([patient[f"patient_token_{i}"] for i in range(self.num_channels)], dtype=torch.float32)
        patient_tokens[3:6] /= self.time_scale
        return patient_tokens  # Add a new dimension for concatenation with event tokens

    def events_to_tensor(self, events):
        if not events:
            return torch.empty((self.num_channels, 0), dtype=torch.float32)
        
        event_tokens = torch.tensor([[event[f"token_{i}"] for event in events] for i in range(self.num_channels)], dtype=torch.float32)
        event_tokens[3:6] /= self.time_scale
        return event_tokens


class NegativeSampler:
    def __init__(self, max_poison_fractions):
        self.max_poison_fractions = max_poison_fractions

    def construct_negative_sample(self, p_tokens, p_age, n_tokens, n_age):
        p_max_select_token_length = int((p_tokens.shape[1] - NUM_PATIENT_TOKENS) * self.max_poison_fractions[0])
        n_tokens = self._ensure_time_order(n_tokens, p_age)
        return self._sample_negative(p_tokens, n_tokens, p_max_select_token_length)

    def _ensure_time_order(self, n_tokens, p_age):
        keep_mask = n_tokens[3, :] < p_age
        keep_mask &= (n_tokens[4, :] < p_age) | (n_tokens[4, :] == float("inf"))
        keep_mask[:NUM_PATIENT_TOKENS] = True
        return n_tokens[:, keep_mask]

    def _sample_negative(self, p_tokens, n_tokens, p_max_select_token_length):
        selected_indices, mapping = self._sample_patient_tokens()
        patient_swap_length = len(selected_indices)

        n_max_select_token_length = int((n_tokens.shape[1] - NUM_PATIENT_TOKENS) * self.max_poison_fractions[1])
        select_token_length = min(
            p_max_select_token_length - patient_swap_length, 
            n_max_select_token_length - patient_swap_length
        )

        if select_token_length > 0:
            new_selected_indices, new_mapping = self._sample_event_tokens(p_tokens, n_tokens, select_token_length)
            selected_indices = torch.cat((selected_indices, new_selected_indices))
            mapping = torch.cat((mapping, new_mapping))

        p_tokens[:, mapping] = n_tokens[:, selected_indices]
        return p_tokens

    def _sample_patient_tokens(self):
        swap = torch.randint(NUM_PATIENT_TOKENS // 2, (1,))
        selected_indices = torch.randperm(NUM_PATIENT_TOKENS)[:swap.item()]
        return selected_indices, selected_indices.clone()

    def _sample_event_tokens(self, p_tokens, n_tokens, select_token_length):
        selected_indices = torch.randperm(n_tokens.shape[1] - NUM_PATIENT_TOKENS)[:select_token_length] + NUM_PATIENT_TOKENS
        mapping = torch.randperm(p_tokens.shape[1] - NUM_PATIENT_TOKENS)[:select_token_length] + NUM_PATIENT_TOKENS
        return selected_indices, mapping


class PatientEventsDataset(Dataset):
    def __init__(self, path, context_length, shuffle=True,
            max_poison_fractions=[0.2, 0.5], threads = 1, lazy_index = False):
        self.context_length = context_length
        self.lazy_index = lazy_index
        self.shuffle = shuffle

        self.repository = PatientRepository(base_path=path, threads=threads, lazy_index=lazy_index)
        self.tensorizer = EventTensorizer(num_channels=NUM_CHANNELS, time_scale=TIME_SCALE)
        self.sampler = NegativeSampler(max_poison_fractions=max_poison_fractions)

        self.mode = "train"
        self.index = []
        self._dataset_length = 0
        self.init_mode(self.mode)

    def init_mode(self, mode: str):
        self.repository.close()
        self.mode = mode
        
        if self.lazy_index:
            # Instant initialization: only read metadata count
            self._dataset_length = self.repository.get_patient_count(mode)
            self.index = []
        else:
            # Eager initialization: pre-load all index rows into RAM
            self.index = self.repository.load_patient_index(mode)
            self._dataset_length = len(self.index)

    def __len__(self) -> int:
        return self._dataset_length

    def get_data_from_database(self, indices: list[int]):
        if self.lazy_index:
            batch_patient_data = self.repository.fetch_patients_by_indices(self.mode, indices)
        else:
            batch_patient_data = [self.index[index] for index in indices]
        batch_events_by_patient = self.repository.fetch_events_by_patients_vectorized(self.mode, batch_patient_data)
        return batch_patient_data, batch_events_by_patient

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.__getitems__([index])
    def __getitems__(self, indices: list[int]) -> list:
        if not indices:
            return []

        batch_patient_data, batch_events_by_patient = self.get_data_from_database(indices)
        new_indices, labels = self.assign_positive_negative_pairs(indices)


        ret_arr = [
            (*self.get_sample(pos, neg, batch_patient_data, batch_events_by_patient),label)
            for pos, neg in new_indices
        ]
        
        return ret_arr 

    def assign_positive_negative_pairs(self, indices):
        # indices = 3k, last 1/3 is negative samples to be paired with the first 1/3 of indices
        if len(indices) == 1:
            return [(indices[0], indices[0])], torch.tensor([1.0], dtype=torch.float32)
        positive = range(len(indices) // 3)
        negative = range(len(indices) // 3, 2 * len(indices) // 3)
        negative_poison = range(2 * len(indices) // 3, len(indices))
        new_indices = [(pos, pos) for pos in positive]
        new_indices += [(pos, neg) for pos, neg in zip(positive, negative_poison)]
        labels = torch.tensor([1] * len(positive) + [0] * len(negative_poison), dtype=torch.float32)
        return new_indices, labels

    def get_sample(self, pos, neg, batch_patient_data, batch_events_by_patient): 
        tokens, patient_age = self._get_patient_tokens(batch_patient_data, batch_events_by_patient, pos)
        
        if neg != pos:
            neg_tokens, neg_patient_age = self._get_patient_tokens(batch_patient_data, batch_events_by_patient, neg)
            tokens = self.sampler.construct_negative_sample(tokens, patient_age, neg_tokens, neg_patient_age)
        tokens, padding_mask = self._enforce_context_length(tokens)
        return tokens, padding_mask
    def _get_patient_tokens(self, batch_patient_data, batch_events_by_patient, index):
        patient = batch_patient_data[index]
        events = batch_events_by_patient[patient["patient_id"]]
        patient_tokens = self.tensorizer.patient_to_tensor(patient)
        patient_age = min(patient_tokens[4, 0].item() - patient_tokens[3, 0].item(), MAX_DATE - patient_tokens[3, 0].item())
        event_tokens = events if isinstance(events, torch.Tensor) else self.tensorizer.events_to_tensor(events)
        tokens = torch.cat((patient_tokens, event_tokens), dim=1)
        return tokens, patient_age

    def get_event_file_groups(self) -> list[list[int]]:
        """Returns row index groups grouped by event_file."""
        # Eager mode: build directly from RAM if index is already populated
        if self.index:
            groups = defaultdict(list)
            for index, row in enumerate(self.index):
                groups[row["event_file"]].append(index)
            return list(groups.values())

        # Lazy mode: execute fast 1-column DuckDB projection query
        return self.repository.fetch_event_file_groups(self.mode)

    def _enforce_context_length(self, tokens):
        _, seq_len = tokens.shape

        if seq_len > self.context_length:
            tokens = self._sample_or_truncate(tokens, NUM_PATIENT_TOKENS)
            padding_mask = torch.zeros(self.context_length, dtype=torch.bool)
        else:
            tokens, padding_mask = self._pad_sequence(tokens)

        return tokens, padding_mask


    def _sample_or_truncate(self, tokens, num_patient_tokens):
        _, seq_len = tokens.shape

        if not self.shuffle:
            return tokens[:, : self.context_length]

        # Preserve patient prefix indices, randomly sample remaining event indices
        patient_idx = torch.arange(num_patient_tokens)
        remaining_pool = torch.arange(num_patient_tokens, seq_len)

        num_samples = self.context_length - num_patient_tokens
        sampled_idx = remaining_pool[torch.randperm(len(remaining_pool))[:num_samples]]

        # Combine and optionally shuffle all selected token positions
        selected_idx = torch.cat((patient_idx, sampled_idx))
        shuffled_idx = selected_idx[torch.randperm(len(selected_idx))]

        return tokens[:, shuffled_idx]


    def _pad_sequence(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pads sequences shorter than context length and generates a padding mask."""
        _, seq_len = tokens.shape

        if self.shuffle:
            tokens = tokens[:, torch.randperm(seq_len)]

        pad_len = self.context_length - seq_len
        tokens = F.pad(tokens, (0, pad_len), mode="constant", value=0)

        padding_mask = torch.zeros(self.context_length, dtype=torch.bool)
        if pad_len > 0:
            padding_mask[-pad_len:] = True

        return tokens, padding_mask
            