"""Shared token layout used by the offline exporter and the PyTorch loader."""

from pathlib import Path

try:
    from .vocab import HealthcareVocab
except ImportError:
    from vocab import HealthcareVocab

PATIENT_DISCRETE_COLUMNS = ("gender", "race", "ethnicity", "birthplace", "city")
PATIENT_FEATURE_COLUMNS = PATIENT_DISCRETE_COLUMNS + ("time", "income")
EVENT_TYPES = {
    "encounter": "encounters", "condition": "conditions", "medication": "medications",
    "procedure": "procedures", "observation": "observations", "immunization": "immunizations",
    "allergy": "allergies", "careplan": "careplans", "imaging_study": "imaging_studies",
}
EVENT_TO_INDEX = {"patients": 0, **{table: i + 1 for i, table in enumerate(EVENT_TYPES.values())}}


class HealthcareTokenizer:
    """Encode database rows into the nine-channel model token format."""

    EVENT_TOKEN_COLUMNS = tuple(f"token_{i}" for i in range(9))

    def __init__(self, vocab_path: str | Path):
        self.vocab = HealthcareVocab(str(vocab_path))

    def encode(self, value):
        return self.vocab.encode(value)

    def tokenize_event(self, event: dict) -> list[float]:
        table = EVENT_TYPES[event["event_type"]]
        value = event.get("value")
        code = self.vocab.encode(None) if event.get("code") is None else self.encode(f"{table}.code.{event['code']}")
        birth = event.get("birth_time", event.get("patient_start_time"))
        start = self._effective_time(event.get("start_time"), event.get("encounter_start_time"))
        stop = self._effective_time(event.get("stop_time"), event.get("encounter_start_time"))
        return [
            code, #discrete value
            float("nan") if value is None else float(value), #continuous value
            0.0 if value is None else 1.0, #is_continuous
            self._age(start, birth), # start age
            self._duration(start, stop), # stop duration
            float(event.get("time_since_previous_event", 0.0)), # time since previous event
            self.encode(table), # table index
            self.encode(f"{table}.code"), # column index
            float(EVENT_TO_INDEX[table]), # event type index decoded
        ]

    def tokenize_patient(self, patient: dict) -> list[list[float]]:
        discrete = [
            self.vocab.encode(None) if patient.get(column) is None
            else self.encode(f"patients.{column}.{patient[column]}")
            for column in PATIENT_DISCRETE_COLUMNS
        ]
        income_present = patient.get("income") is not None
        income = float("nan") if not income_present else float(patient["income"])
        # Income is continuous: keep token_0 as the categorical/missing-value
        # channel and put the actual amount in token_1.
        values = discrete + [self.vocab.encode(None), self.vocab.encode(None)]
        columns = [self.encode(f"patients.{column}") for column in PATIENT_FEATURE_COLUMNS]
        start = self._time(patient.get("patient_start_time"))
        stop = self._duration(patient.get("patient_start_time"), patient.get("patient_stop_time"))
        return [values, [float("nan")] * 6 + [income], [0.0] * 6 + [float(income_present)],
            [start] * 7, [stop] * 7, [0.0] * 7, [self.encode("patients")] * 7,
            columns, [EVENT_TO_INDEX["patients"]] * 7]

    @staticmethod
    def _effective_time(value, encounter_start):
        if value is None or encounter_start is None:
            return value
        return max(value, encounter_start)

    @classmethod
    def _age(cls, value, birth) -> float:
        if value is None or birth is None:
            return float("inf")
        return value.timestamp() - birth.timestamp()

    @classmethod
    def _duration(cls, start, stop) -> float:
        if start is None or stop is None:
            return float("inf")
        return max(stop.timestamp() - start.timestamp(), 0.0)

    @staticmethod
    def _time(value) -> float:
        if value is None or str(value) == "NaT":
            return float("inf")
        return value.timestamp()

    @staticmethod
    def db_key(value) -> str:
        return "__NULL__" if value is None else str(value)

    def db_vocabulary(self):
        return [(self.db_key(token), token_id) for token, token_id in self.vocab.str_to_id.items()]