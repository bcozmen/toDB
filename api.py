import json
import math
import random
from pathlib import Path
from collections.abc import Mapping
from numbers import Real
from fastapi import FastAPI, HTTPException
import uvicorn
from src.dataloader import PatientRepository

try:
    import numpy as np
except ImportError:  # pragma: no cover - NumPy is normally installed with torch.
    np = None

try:
    import torch
except ImportError:  # pragma: no cover - only needed when tensor fields are returned.
    torch = None

app = FastAPI()

SCHEMA_FILE = Path("/home/baris/database_transformer/dataset/schema.json")
DB_PATH = Path("/home/baris/database_transformer/dataset/ml")

patient_repo = PatientRepository(str(DB_PATH), num_channels = 9, lazy_index = True, threads= 8)
patient_repo.mode = "test"
patient_repo._dataset_count = patient_repo.get_patient_count(patient_repo.mode)

from fastapi.middleware.cors import CORSMiddleware

allow_origins = ["http://localhost:5173", "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/schema")
def get_schema():
    if not SCHEMA_FILE.is_file():
        raise HTTPException(status_code=404, detail="schema.json file not found")
    
    try:
        with open(SCHEMA_FILE, "r", encoding="utf-8") as f:
            raw_schema = json.load(f)

        # Keep the API response aligned with the frontend's table/column model.
        return {
            "tables": [
                {
                    "name": table_name,
                    "dictionary": columns.get("dictionary", {}),
                    "columns": [
                        {
                            "column_name": column_name,
                            "column_type": column.get("column_type", ""),
                            "primary_key": column.get("primary_key", False),
                            "foreign_key": _normalize_foreign_key(
                                column.get("foreign_key")
                            ),
                        }
                        for column_name, column in columns.items()
                        if column_name != "dictionary"
                    ],
                }
                for table_name, columns in raw_schema.items()
            ]
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON format in schema.json")

def sanitize_nan(data):
    """Recursively replaces NaN float/numpy values with None for JSON compliance."""
    if isinstance(data, float) and math.isnan(data):
        return None
    if isinstance(data, dict):
        return {k: sanitize_nan(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_nan(item) for item in data]
    return data

def sanitize_floats(data):
    """Recursively replaces NaN, inf, and -inf float values with None (null in JSON)."""
    # Convert tensor/array containers before walking their values. FastAPI's
    # JSON encoder cannot serialize non-finite values nested inside them.
    if torch is not None and isinstance(data, torch.Tensor):
        return sanitize_floats(data.detach().cpu().tolist())
    if np is not None and isinstance(data, np.ndarray):
        return sanitize_floats(data.tolist())
    if isinstance(data, (list, tuple)):
        return [sanitize_floats(item) for item in data]
    if isinstance(data, Mapping):
        return {k: sanitize_floats(v) for k, v in data.items()}

    # Handle standard Python floats and NumPy scalar values.
    if isinstance(data, Real) or (np is not None and isinstance(data, np.generic)):
        try:
            value = data.item() if np is not None and isinstance(data, np.generic) else data
            if isinstance(value, Real) and not math.isfinite(value):
                return None
            return value
        except (TypeError, ValueError):
            pass

    return data

@app.get("/random_patient")
def get_random_patient():
    try:
        rnd_index = random.randint(0, patient_repo._dataset_count - 1)
        patient_data = patient_repo.fetch_patients_by_indices(patient_repo.mode, [rnd_index])[0]
        batch_events_by_patient = patient_repo.fetch_events_by_patients(patient_repo.mode, patient_data)
        new_dict = {
            'patient': patient_data,
            'events': batch_events_by_patient
        }
        #return json response
        return sanitize_floats(new_dict)
    except IndexError:
        raise HTTPException(status_code=404, detail="No patients found in the dataset")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def _normalize_foreign_key(value):
    if not value:
        return None

    if isinstance(value, str) and "." in value:
        table, column = value.split(".", 1)
        return {"table": table, "column": column}

    return value

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)