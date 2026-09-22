import duckdb
import numpy as np
import torch
from collections import defaultdict

class PatientRepository:
    """Handles connection management and queries for patient/event parquet files."""
    
    def __init__(self, base_path, num_channels, threads=1, lazy_index=False):
        self.num_channels = num_channels
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

    def patient_to_tensor(self, patient):
        patient_tokens = torch.from_numpy(
            np.stack([patient[f"patient_token_{i}"] for i in range(self.num_channels)])
        ).to(dtype=torch.float32)
        return patient_tokens  # Add a new dimension for concatenation with event tokens

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
            SELECT patient_id, {', '.join(f'token_{i}' for i in range(self.num_channels))}
            FROM read_parquet('{file_path}')
            WHERE patient_id IN ({placeholders})
            ORDER BY patient_id, "index"
        """
        np_dict = self._get_connection().execute(query, patient_ids).fetchnumpy()

        if not np_dict or len(np_dict["patient_id"]) == 0:
            return {}

        p_ids = np.asarray(np_dict["patient_id"])
        token_columns = np.column_stack(
            [np.asarray(np_dict[f"token_{i}"]) for i in range(self.num_channels)]
        )

        # The query orders by patient_id, so build contiguous slices instead of
        # scanning the complete batch once for every patient.  The old
        # ``p_ids == pid`` comprehension is O(number_of_patients * rows) and
        # becomes a major bottleneck for large event files.
        boundaries = np.flatnonzero(p_ids[1:] != p_ids[:-1]) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, len(p_ids)]
        events_by_patient = {
            (p_ids[start].item() if isinstance(p_ids[start], np.generic) else p_ids[start]):
            torch.from_numpy(token_columns[start:end].T).float()
            for start, end in zip(starts, ends)
        }
        return events_by_patient
        