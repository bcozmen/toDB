import torch
import duckdb
import torch
from pathlib import Path
from helper.vocab import HealthcareVocab


DATASET_ROOT = "/home/baris/database_transformer/dataset"
DATABASE_PATH = f"{DATASET_ROOT}/healthcare.duckdb"
VOCAB_PATH = f"{DATASET_ROOT}/ml/vocab.pt"

VOCAB_CATEGORICAL_COLUMNS = {
    "patients": ['gender', 'race', 'ethnicity', 'birthplace', 'city'],
    "encounters": ['code'],
    "conditions": ['code'],
    "medications": ['code'],
    "procedures": ['code'],
    "observations": ['code', 'units'],
    "immunizations": ['code'],
    "allergies": ['code'],
    "careplans": ['code'],
    "imaging_studies": ['bodysite_code'],
}

VOCAB_NUMERIC_COLUMNS = {
    "patients": ['time', 'income'],
    "observations": ['value'],
}
#Discrete Timestamp, Continuous
SPECIAL_TOKENS = [None, torch.inf, torch.nan]


out_dir = Path(VOCAB_PATH).parent
out_dir.mkdir(parents=True, exist_ok=True)


con = duckdb.connect(str(DATABASE_PATH), read_only=True)
dictionary = HealthcareVocab()
dictionary.add(SPECIAL_TOKENS)
for table_name, columns in VOCAB_CATEGORICAL_COLUMNS.items():
    dictionary.add(table_name)
    for column in columns:
        query = f"""
        SELECT DISTINCT CAST("{column}" AS VARCHAR)
        FROM "{table_name}"
        WHERE "{column}" IS NOT NULL
        ORDER BY 1
        """

        if table_name == "imaging_studies" and column == "bodysite_code":
            column = "code"

        unique_values = [table_name + "." + column + "." + row[0] for row in con.execute(query).fetchall()]

        dictionary.add(unique_values)
        dictionary.add(table_name + "." + column)

for table_name, columns in VOCAB_NUMERIC_COLUMNS.items():
    dictionary.add(table_name)
    for column in columns:
        dictionary.add(table_name + "." + column)


dictionary.save(VOCAB_PATH)