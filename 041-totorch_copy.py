from pathlib import Path
import duckdb
from helper.tokenizer import HealthcareTokenizer

DATASET_PATH = Path("/home/baris/database_transformer/dataset")
ML_PATH = DATASET_PATH / "ml"
DATABASE_PATH = DATASET_PATH / "healthcare.duckdb"
VOCAB_PATH = ML_PATH / "vocab.pt"

SPLITS = {
    "train": range(0, 16),
    "val": range(16, 18),
    "test": range(18, 20),
}

# Table metadata for dynamically building the unified view
EVENT_TABLES = [
    # (table_name, event_type, code_col, value_col, units_col, stop_time_col, extra_filter)
    ("encounters", "encounter", "code", "NULL", "NULL", "stop_time", None),
    ("conditions", "condition", "code", "NULL", "NULL", "stop_time", None),
    ("medications", "medication", "code", "NULL", "NULL", "stop_time", None),
    ("procedures", "procedure", "code", "NULL", "NULL", "stop_time", None),
    ("observations", "observation", "code", "value", "units", "NULL::TIMESTAMP", "type = 'numeric'"),
    ("immunizations", "immunization", "code", "NULL", "NULL", "NULL::TIMESTAMP", None),
    ("allergies", "allergy", "code", "NULL", "NULL", "stop_time", None),
    ("careplans", "careplan", "code", "NULL", "NULL", "stop_time", None),
    ("imaging_studies", "imaging_study", "bodysite_code", "NULL", "NULL", "NULL::TIMESTAMP", None),
]


def create_events_view(con: duckdb.DuckDBPyConnection) -> None:
    """Builds a temporary view to avoid re-parsing identical CTEs across buckets."""
    queries = []
    for tbl, event_type, code, val, units, stop_time, extra in EVENT_TABLES:
        where_clause = f"WHERE patient_id IS NOT NULL AND {extra}" if extra else "WHERE patient_id IS NOT NULL"
        queries.append(f"""
            SELECT
                patient_id, encounter_id, start_time,
                {stop_time} AS stop_time,
                '{event_type}' AS event_type,
                {code}::VARCHAR AS code,
                {val}::DOUBLE AS value,
                {units}::VARCHAR AS units
            FROM {tbl}
            {where_clause}
        """)

    union_sql = " UNION ALL ".join(queries)
    con.execute("DROP VIEW IF EXISTS unified_events")
    con.execute(f"CREATE TEMP VIEW unified_events AS {union_sql}")


def create_token_vocab(con, tokenizer):
    con.execute("CREATE OR REPLACE TEMP TABLE token_vocab (token_key VARCHAR, token_id BIGINT)")
    con.executemany("INSERT INTO token_vocab VALUES (?, ?)", tokenizer.db_vocabulary())


def create_tokenized_events_view(con):
    """Add normalized times and numeric token channels once before export."""
    con.execute("DROP VIEW IF EXISTS tokenized_events")
    con.execute("""
        CREATE TEMP VIEW tokenized_events AS
        WITH normalized AS (
            SELECT e.*,
                CASE WHEN encounter.start_time IS NOT NULL
                          AND e.start_time < encounter.start_time
                     THEN encounter.start_time ELSE e.start_time END AS normalized_start_time,
                CASE WHEN encounter.start_time IS NOT NULL
                          AND e.stop_time < encounter.start_time
                     THEN encounter.start_time ELSE e.stop_time END AS normalized_stop_time,
                patient.start_time AS birth_time
            FROM unified_events e
            LEFT JOIN patients patient ON patient.patient_id = e.patient_id
            LEFT JOIN encounters encounter ON encounter.encounter_id = e.encounter_id
        ), ordered AS (
            SELECT n.*,
                LAG(normalized_start_time) OVER (
                    PARTITION BY patient_id
                    ORDER BY CAST(normalized_start_time AS DATE) ASC NULLS LAST,
                        CASE WHEN event_type = 'encounter' THEN 0 ELSE 1 END,
                        normalized_start_time ASC NULLS LAST,
                        normalized_stop_time ASC NULLS LAST,
                        encounter_id ASC NULLS LAST,
                        event_type ASC,
                        code ASC
                ) AS previous_start_time
            FROM normalized n
        )
        SELECT o.* EXCLUDE (normalized_start_time, normalized_stop_time,
                            birth_time, previous_start_time),
            code_v.token_id AS token_0,
            CASE WHEN o.value IS NULL THEN 'nan'::DOUBLE ELSE o.value END AS token_1,
            CASE WHEN o.value IS NULL THEN 0.0 ELSE 1.0 END AS token_2,
            CASE WHEN o.normalized_start_time IS NULL OR o.birth_time IS NULL THEN 'inf'::DOUBLE
                 ELSE epoch(o.normalized_start_time) - epoch(o.birth_time) END AS token_3,
              CASE WHEN o.normalized_stop_time IS NULL OR o.normalized_start_time IS NULL
                      OR o.birth_time IS NULL THEN 'inf'::DOUBLE
                  ELSE GREATEST(epoch(o.normalized_stop_time) - epoch(o.normalized_start_time), 0.0)
              END AS token_4,
              CASE WHEN o.previous_start_time IS NULL OR o.normalized_start_time IS NULL
                      THEN 0.0
                  ELSE epoch(o.normalized_start_time) - epoch(o.previous_start_time) END AS token_5,
              table_v.token_id AS token_6, column_v.token_id AS token_7,
              CASE o.event_type WHEN 'encounter' THEN 1 WHEN 'condition' THEN 2 WHEN 'medication' THEN 3
                 WHEN 'procedure' THEN 4 WHEN 'observation' THEN 5 WHEN 'immunization' THEN 6
                 WHEN 'allergy' THEN 7 WHEN 'careplan' THEN 8 WHEN 'imaging_study' THEN 9 END::DOUBLE AS token_8
        FROM ordered o
        LEFT JOIN token_vocab code_v ON code_v.token_key =
            CASE WHEN o.code IS NULL THEN '__NULL__' ELSE
                (CASE o.event_type WHEN 'encounter' THEN 'encounters' WHEN 'condition' THEN 'conditions'
                 WHEN 'medication' THEN 'medications' WHEN 'procedure' THEN 'procedures'
                 WHEN 'observation' THEN 'observations' WHEN 'immunization' THEN 'immunizations'
                 WHEN 'allergy' THEN 'allergies' WHEN 'careplan' THEN 'careplans'
                  WHEN 'imaging_study' THEN 'imaging_studies' END) || '.code.' || o.code
            END
        LEFT JOIN token_vocab table_v ON table_v.token_key =
              (CASE o.event_type WHEN 'encounter' THEN 'encounters' WHEN 'condition' THEN 'conditions'
             WHEN 'medication' THEN 'medications' WHEN 'procedure' THEN 'procedures'
             WHEN 'observation' THEN 'observations' WHEN 'immunization' THEN 'immunizations'
             WHEN 'allergy' THEN 'allergies' WHEN 'careplan' THEN 'careplans'
             WHEN 'imaging_study' THEN 'imaging_studies' END)
        LEFT JOIN token_vocab column_v ON column_v.token_key =
            (CASE o.event_type WHEN 'encounter' THEN 'encounters' WHEN 'condition' THEN 'conditions'
             WHEN 'medication' THEN 'medications' WHEN 'procedure' THEN 'procedures'
             WHEN 'observation' THEN 'observations' WHEN 'immunization' THEN 'immunizations'
             WHEN 'allergy' THEN 'allergies' WHEN 'careplan' THEN 'careplans'
             WHEN 'imaging_study' THEN 'imaging_studies' END) || '.code'
    """)


def export_events(con: duckdb.DuckDBPyConnection) -> None:
    create_tokenized_events_view(con)

    for split, buckets in SPLITS.items():
        out_dir = ML_PATH / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for bucket in buckets:
            out_path = out_dir / f"part-{bucket:02d}.parquet"
            out_path.unlink(missing_ok=True)

            con.execute(f"""
                COPY (
                    SELECT
                        e.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY patient_id
                            ORDER BY
                                token_3 ASC NULLS LAST,
                                CASE WHEN event_type = 'encounter' THEN 0 ELSE 1 END,
                                token_3 ASC NULLS LAST,
                                token_4 ASC NULLS LAST,
                                encounter_id ASC NULLS LAST,
                                event_type ASC,
                                code ASC
                        ) - 1 AS "index"
                    FROM tokenized_events e
                    WHERE abs(hash(patient_id)) % 20 = {bucket}
                    ORDER BY
                        patient_id,
                        "index"
                )
                TO '{out_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000);
            """)


def build_patient_index(con: duckdb.DuckDBPyConnection, split: str) -> None:
    split_dir = ML_PATH / split
    index_path = split_dir / "patient_index.parquet"
    files = split_dir / "part-*.parquet"

    index_path.unlink(missing_ok=True)

    con.execute(f"""
        COPY (
            SELECT
                e.patient_id,
                p.start_time AS patient_start_time,
                p.stop_time AS patient_stop_time,
                p.gender, p.race, p.ethnicity, p.birthplace, p.city, p.income,
                regexp_extract(filename, 'part-([0-9]+)[.]parquet$', 1)::INTEGER AS bucket,
                regexp_extract(filename, '(part-[0-9]+[.]parquet)$', 1) AS event_file,
                MIN(e.start_time) AS first_event_time,
                MAX(e.start_time) AS last_event_time,
                COUNT(*) AS n_events,
                list_value(
                    gender_v.token_id::DOUBLE, race_v.token_id::DOUBLE,
                    ethnicity_v.token_id::DOUBLE, birthplace_v.token_id::DOUBLE,
                    city_v.token_id::DOUBLE, null_v.token_id::DOUBLE
                ) AS patient_token_0,
                list_value('nan'::DOUBLE, 'nan'::DOUBLE, 'nan'::DOUBLE,
                    'nan'::DOUBLE, 'nan'::DOUBLE,
                    CASE WHEN p.income IS NULL THEN 'nan'::DOUBLE ELSE p.income END
                ) AS patient_token_1,
                list_value(0.0, 0.0, 0.0, 0.0, 0.0,
                    CASE WHEN p.income IS NULL THEN 0.0 ELSE 1.0 END
                ) AS patient_token_2,
                list_value(
                    CASE WHEN p.start_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.start_time) END,
                    CASE WHEN p.start_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.start_time) END,
                    CASE WHEN p.start_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.start_time) END,
                    CASE WHEN p.start_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.start_time) END,
                    CASE WHEN p.start_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.start_time) END,
                    CASE WHEN p.start_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.start_time) END
                ) AS patient_token_3,
                list_value(
                    CASE WHEN p.stop_time IS NULL OR p.start_time IS NULL THEN 'inf'::DOUBLE
                        ELSE GREATEST(epoch(p.stop_time) - epoch(p.start_time), 0.0) END,
                    CASE WHEN p.stop_time IS NULL OR p.start_time IS NULL THEN 'inf'::DOUBLE
                        ELSE GREATEST(epoch(p.stop_time) - epoch(p.start_time), 0.0) END,
                    CASE WHEN p.stop_time IS NULL OR p.start_time IS NULL THEN 'inf'::DOUBLE
                        ELSE GREATEST(epoch(p.stop_time) - epoch(p.start_time), 0.0) END,
                    CASE WHEN p.stop_time IS NULL OR p.start_time IS NULL THEN 'inf'::DOUBLE
                        ELSE GREATEST(epoch(p.stop_time) - epoch(p.start_time), 0.0) END,
                    CASE WHEN p.stop_time IS NULL OR p.start_time IS NULL THEN 'inf'::DOUBLE
                        ELSE GREATEST(epoch(p.stop_time) - epoch(p.start_time), 0.0) END,
                    CASE WHEN p.stop_time IS NULL OR p.start_time IS NULL THEN 'inf'::DOUBLE
                        ELSE GREATEST(epoch(p.stop_time) - epoch(p.start_time), 0.0) END
                ) AS patient_token_4,
                 list_value(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) AS patient_token_5,
                list_value(patients_v.token_id::DOUBLE, patients_v.token_id::DOUBLE,
                    patients_v.token_id::DOUBLE, patients_v.token_id::DOUBLE,
                    patients_v.token_id::DOUBLE, patients_v.token_id::DOUBLE
                 ) AS patient_token_6,
                list_value(gender_column_v.token_id::DOUBLE, race_column_v.token_id::DOUBLE,
                    ethnicity_column_v.token_id::DOUBLE, birthplace_column_v.token_id::DOUBLE,
                    city_column_v.token_id::DOUBLE, income_column_v.token_id::DOUBLE
                 ) AS patient_token_7,
                 list_value(0.0, 0.0, 0.0, 0.0, 0.0, 0.0) AS patient_token_8
            FROM read_parquet('{files.as_posix()}', filename=true) e
            LEFT JOIN patients p ON p.patient_id = e.patient_id
            LEFT JOIN token_vocab gender_v ON gender_v.token_key =
                CASE WHEN p.gender IS NULL THEN '__NULL__' ELSE 'patients.gender.' || p.gender END
            LEFT JOIN token_vocab race_v ON race_v.token_key =
                CASE WHEN p.race IS NULL THEN '__NULL__' ELSE 'patients.race.' || p.race END
            LEFT JOIN token_vocab ethnicity_v ON ethnicity_v.token_key =
                CASE WHEN p.ethnicity IS NULL THEN '__NULL__' ELSE 'patients.ethnicity.' || p.ethnicity END
            LEFT JOIN token_vocab birthplace_v ON birthplace_v.token_key =
                CASE WHEN p.birthplace IS NULL THEN '__NULL__' ELSE 'patients.birthplace.' || p.birthplace END
            LEFT JOIN token_vocab city_v ON city_v.token_key =
                CASE WHEN p.city IS NULL THEN '__NULL__' ELSE 'patients.city.' || p.city END
            LEFT JOIN token_vocab null_v ON null_v.token_key = '__NULL__'
            LEFT JOIN token_vocab patients_v ON patients_v.token_key = 'patients'
            LEFT JOIN token_vocab gender_column_v ON gender_column_v.token_key = 'patients.gender'
            LEFT JOIN token_vocab race_column_v ON race_column_v.token_key = 'patients.race'
            LEFT JOIN token_vocab ethnicity_column_v ON ethnicity_column_v.token_key = 'patients.ethnicity'
            LEFT JOIN token_vocab birthplace_column_v ON birthplace_column_v.token_key = 'patients.birthplace'
            LEFT JOIN token_vocab city_column_v ON city_column_v.token_key = 'patients.city'
            LEFT JOIN token_vocab income_column_v ON income_column_v.token_key = 'patients.income'
            GROUP BY
                e.patient_id, p.start_time, p.stop_time, p.gender, p.race,
                p.ethnicity, p.birthplace, p.city, p.income, bucket, event_file,
                gender_v.token_id, race_v.token_id, ethnicity_v.token_id,
                birthplace_v.token_id, city_v.token_id, null_v.token_id,
                patients_v.token_id, gender_column_v.token_id, race_column_v.token_id,
                ethnicity_column_v.token_id, birthplace_column_v.token_id,
                city_column_v.token_id, income_column_v.token_id
        )
        TO '{index_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000);
    """)


def main() -> None:
    ML_PATH.mkdir(parents=True, exist_ok=True)
    tokenizer = HealthcareTokenizer(VOCAB_PATH)

    with duckdb.connect(DATABASE_PATH.as_posix()) as con:
        con.execute("SET memory_limit = '40GB'")
        con.execute("SET threads = 12")
        con.execute("SET preserve_insertion_order = false")
        con.execute(f"SET temp_directory = '{(ML_PATH / 'duckdb_tmp').as_posix()}'")

        create_events_view(con)
        create_token_vocab(con, tokenizer)
        export_events(con)

        for split in SPLITS:
            build_patient_index(con, split)


if __name__ == "__main__":
    main()