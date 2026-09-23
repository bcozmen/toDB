import duckdb
import pandas as pd
import json

def get_unique_codes(table_name):
    code = "code" if table_name != "imaging_studies" else "bodysite_code"
    description = "description" if table_name != "imaging_studies" else "bodysite_description"
    query = f"""
                SELECT DISTINCT {code} AS code, {description} AS description
                FROM {table_name}
                WHERE {code} IS NOT NULL
                ORDER BY code
            """
    df = con.execute(query).fetchdf()
    mapping = dict(zip(df['code'], df['description']))
    return mapping

def get_unique_patients():
    columns = ["gender", "race", "ethnicity", "birthplace", "city"]
    mappings = {}
    for column in columns:
        query = f"""
                    SELECT DISTINCT {column}
                    FROM patients
                    WHERE {column} IS NOT NULL
                    ORDER BY {column}
                """
        df = con.execute(query).fetchdf()
        col = df[column].tolist()
        mappings[column] = col
    return mappings

def build_dictionary(row):
    column = {}
    for k, v in row.items():
        if k == "null" and v == "YES":
            continue
        if k in ("key", "default", "extra") and (v is None or pd.isna(v) or str(v) == "None"):
            continue
        column[k] = v
    return column

def set_keys(column, column_name, table_name):
    primary_key = False
    foreign_key = None

    if column_name == "patient_id":
        if table_name == "patients":
            primary_key = True
        else:
            foreign_key = {
                'table': 'patients',
                'column': 'patient_id'
            }
    elif column_name == "encounter_id":
        if table_name == "encounters":
            primary_key = True
        else:
            foreign_key = {
                'table': 'encounters',
                'column': 'encounter_id'
            }

    column["primary_key"] = primary_key
    column["foreign_key"] = foreign_key
    return column

def parse_table(table_name, df_scheme):
    columns = {}

    for _, row in df_scheme.iterrows():
        column_name = row["column_name"]
        if any(keyword in column_name for keyword in EXCLUDE_KEYWORDS):
            continue
        column_name = column_name.replace("bodysite_","")

        # Build dictionary, skipping unwanted metadata
        column = build_dictionary(row)

        # Set primary/foreign keys
        column = set_keys(column, column_name, table_name)

        # Add the processed column to the columns dictionary
        columns[column_name] = column

    return columns



PATH = "/home/baris/database_transformer/dataset/"
con = duckdb.connect(f"{PATH}/healthcare.duckdb", read_only=True)
schema = {}

# Keywords to exclude from column processing
EXCLUDE_KEYWORDS = ("modality",  "procedure_code", "system", "reason", "quantity", "type", "category", \
                    "reaction", "description_", "severity", "encounter_class")

tables = con.execute("SHOW TABLES").fetchall()
for table in tables:
    table_name = table[0]
    df_scheme = con.execute(f"DESCRIBE {table_name}").df()

    schema[table_name] = parse_table(table_name, df_scheme)

    if table_name == "patients":
        dictionary = get_unique_patients()
    else:
        dictionary = get_unique_codes(table_name)
    schema[table_name]["dictionary"] = dictionary

# Save as json file
with open(f"{PATH}/schema.json", "w") as f:
     json.dump(schema, f)
# Print schema representation
for table, columns in schema.items():
    print(f"Table: {table}")
    for column_name, column in columns.items():
        print(f"  Column: {column_name}")
        for k, v in column.items():
            print(f"    {k}: {v}")