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


# Time is normalized by ten years in the dataset loader.
DEFAULT_FUTURE_HORIZONS = ["1 week", "1 month", "6 months", "1 year", "5 years", "10 years"]


from predictor import model
@app.post("/ai_insights")
def get_ai_insights(request: dict):
    try:
        patient_data = request.get("patient")
        events_data = request.get("events")
        if patient_data is None or events_data is None:
            raise HTTPException(status_code=400, detail="Missing patient or events data in the request")

        patient = patient_repo.patient_to_tensor(patient_data)
        events = patient_repo.events_to_tensor(events_data)


        tokens = torch.cat([patient, events], dim=1)  # Concatenate patient and events tensors along the sequence dimension

        #add a extra token dimension to the tokens tensor to match the model's expected input shape
        tokens = torch.cat([tokens, torch.zeros((2, tokens.shape[1]), device=tokens.device)], dim=0)  # Add a token dimension
        padding_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)  # No padding in this example
        tokens = tokens.unsqueeze(0)  # Add batch dimension
        padding_mask = padding_mask.unsqueeze(0)  # Add batch dimension

        sequence_length = tokens.shape[-1]
        if sequence_length > model.context_size or sequence_length < 2:
            raise HTTPException(status_code=400, detail=f"Sequence length {sequence_length} is out of bounds for the model's context size {model.context_size}")
        with torch.no_grad():
            # Float16 autocast can overflow intermediate activations even
            # when the checkpoint parameters are finite, producing NaN logits.
            # Keep API inference in float32; the model was trained with AMP.
            with torch.autocast(device_type='cuda', enabled=False):
                class_logits, next_event, future_encounter = model((tokens, padding_mask))  # Add batch dimension
        print("Class logits shape:", class_logits.shape)
        hazard_logits, code_logits = future_encounter
        print("Hazard logits shape:", hazard_logits.shape)
        print("Code logits shape:", code_logits.shape)
        time_params, table_logits, code_logits = next_event
        print("Time params shape:", time_params.shape)
        print("Table logits shape:", table_logits.shape)
        print("Code logits shape:", code_logits.shape)
        
        raw_hazard = hazard_logits[0, -1].float()
        if not torch.isfinite(raw_hazard).all():
            raise HTTPException(status_code=500, detail="Model returned non-finite future hazard logits")

        # `hazard[k]` is conditional on surviving all earlier intervals.
        # Convert logits into both interval-event and cumulative probabilities.
        interval_hazard = torch.sigmoid(raw_hazard)
        survival_before = torch.cumprod(
            1.0 - interval_hazard, dim=-1
        ).roll(1, dims=-1)
        survival_before[0] = 1.0
        interval_probability = survival_before * interval_hazard
        cumulative_probability = 1.0 - torch.cumprod(
            1.0 - interval_hazard, dim=-1
        )
        print("Interval event probability:", interval_probability.tolist())
        print("Cumulative event probability:", cumulative_probability.tolist())
        future_code = np.random.randint(0, 5, size=6)  # Mock code predictions
        # Here you would implement your AI insights logic based on the patient and events data.
        # For demonstration purposes, we'll return a mock response.
        next_time = "2024-07-01T12:00:00Z"  # Mock next time prediction
        next_table = "observations"  # Mock next table prediction
        next_code = "123456"
        ai_insights = {
            # Preserve the conditional hazards and expose the two useful
            # probability interpretations separately.
            "future_hazard": cumulative_probability.tolist(),
            "future_code": future_code.tolist(),  # Mock code predictions
            "future_horizon": DEFAULT_FUTURE_HORIZONS,  # Mock horizons
            "next_time": next_time,
            "next_table": next_table,
            "next_code": next_code
        }
        return ai_insights
    except HTTPException:
        raise
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