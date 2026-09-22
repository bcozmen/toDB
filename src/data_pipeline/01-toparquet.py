import duckdb
from __init__ import DATASET_PATH

CSV_PATH = DATASET_PATH / "csv"
PARQUET_PATH = DATASET_PATH / "parquet"


EXPORTS = {
    "patients": {
        "columns": """
            Id AS patient_id,
            TRY_CAST(BIRTHDATE AS TIMESTAMP) AS start_time,
            TRY_CAST(DEATHDATE AS TIMESTAMP) AS stop_time,
            GENDER AS gender,
            RACE AS race,
            ETHNICITY AS ethnicity,
            BIRTHPLACE AS birthplace,
            CITY AS city,
            TRY_CAST(INCOME AS DOUBLE) AS income
        """,
        "source": "patients.csv",
        "where": "Id IS NOT NULL",
    },
    "encounters": {
        "columns": """
            Id AS encounter_id,
            PATIENT AS patient_id,
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            ENCOUNTERCLASS AS encounter_class,
            CODE AS code,
            DESCRIPTION AS description,
            REASONCODE AS reason_code,
            REASONDESCRIPTION AS reason_description
        """,
        "source": "encounters.csv",
        "where": "Id IS NOT NULL AND PATIENT IS NOT NULL",
    },
    "conditions": {
        "columns": """
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            SYSTEM AS system,
            CODE AS code,
            DESCRIPTION AS description
        """,
        "source": "conditions.csv",
    },
    "medications": {
        "columns": """
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CODE AS code,
            DESCRIPTION AS description,
            REASONCODE AS reason_code,
            REASONDESCRIPTION AS reason_description
        """,
        "source": "medications.csv",
    },
    "observations": {
        "columns": """
            TRY_CAST(DATE AS TIMESTAMP) AS start_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CATEGORY AS category,
            CODE AS code,
            DESCRIPTION AS description,
            VALUE AS value,
            UNITS AS units,
            TYPE AS type
        """,
        "source": "observations.csv",
    },
    "procedures": {
        "columns": """
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            SYSTEM AS system,
            CODE AS code,
            DESCRIPTION AS description,
            REASONCODE AS reason_code,
            REASONDESCRIPTION AS reason_description
        """,
        "source": "procedures.csv",
    },
    "allergies": {
        "columns": """
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CODE AS code,
            SYSTEM AS system,
            DESCRIPTION AS description,
            TYPE AS type,
            CATEGORY AS category,
            REACTION1 AS reaction_1,
            DESCRIPTION1 AS description_1,
            REACTION2 AS reaction_2,
            DESCRIPTION2 AS description_2,
            SEVERITY2 AS severity_2
        """,
        "source": "allergies.csv",
    },
    "careplans": {
        "columns": """
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CODE AS code,
            DESCRIPTION AS description,
            REASONCODE AS reason_code,
            REASONDESCRIPTION AS reason_description
        """,
        "source": "careplans.csv",
    },
    "imaging_studies": {
        "columns": """
            TRY_CAST(DATE AS TIMESTAMP) AS start_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            BODYSITE_CODE AS bodysite_code,
            BODYSITE_DESCRIPTION AS bodysite_description,
            MODALITY_CODE AS modality_code,
            MODALITY_DESCRIPTION AS modality_description,
            PROCEDURE_CODE AS procedure_code
        """,
        "source": "imaging_studies.csv",
    },
    "devices": {
        "columns": """
            TRY_CAST(START AS TIMESTAMP) AS start_time,
            TRY_CAST(STOP AS TIMESTAMP) AS stop_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CODE AS code,
            DESCRIPTION AS description,
        """,
        "source": "devices.csv",
    },
    "supplies": {
        "columns": """
            TRY_CAST(DATE AS TIMESTAMP) AS start_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CODE AS code,
            DESCRIPTION AS description,
            QUANTITY AS quantity
        """,
        "source": "supplies.csv",
    },
    "immunizations": {
        "columns": """
            TRY_CAST(DATE AS TIMESTAMP) AS start_time,
            PATIENT AS patient_id,
            ENCOUNTER AS encounter_id,
            CODE AS code,
            DESCRIPTION AS description
        """,
        "source": "immunizations.csv",
    },
}


def build_export_query(table_name, export):
    source = CSV_PATH / export["source"]
    destination = PARQUET_PATH / f"{table_name}.parquet"
    where_clause = export.get(
        "where",
        "PATIENT IS NOT NULL AND ENCOUNTER IS NOT NULL",
    )

    return f"""
        COPY (
            SELECT
                {export["columns"]}
            FROM read_csv('{source}', auto_detect = true)
            WHERE {where_clause}
        )
        TO '{destination}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """


def main():
    PARQUET_PATH.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as con:
        con.execute("SET threads = 24")
        con.execute("SET preserve_insertion_order = false")
        for table_name, export in EXPORTS.items():
            con.execute(build_export_query(table_name, export))
            print(f"Exported {table_name}")


if __name__ == "__main__":
    main()
