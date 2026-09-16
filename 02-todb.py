from pathlib import Path
import duckdb


DATASET_PATH = Path("/home/baris/database_transformer/dataset")
PARQUET_PATH = DATASET_PATH / "parquet"
DATABASE_PATH = DATASET_PATH / "healthcare.duckdb"


TABLES = [
    "patients",
    "encounters",
    "conditions",
    "medications",
    "observations",
    "procedures",
    "allergies",
    "careplans",
    "imaging_studies",
    "devices",
    "supplies",
    "immunizations",
]


def main():
    with duckdb.connect(str(DATABASE_PATH)) as con:
        for table in TABLES:
            parquet = PARQUET_PATH / f"{table}.parquet"

            con.execute(f"""
                CREATE OR REPLACE VIEW {table} AS
                SELECT *
                FROM read_parquet('{parquet}');
            """)

            print(f"Created view {table}")

    print(f"Database: {DATABASE_PATH}")


if __name__ == "__main__":
    main()
