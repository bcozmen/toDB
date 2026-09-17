from pathlib import Path
import os
import shutil
import time
import duckdb
import pandas as pd

from helper.tokenizer import (
    EVENT_TO_INDEX,
    EVENT_TYPES,
    PATIENT_DISCRETE_COLUMNS,
    PATIENT_FEATURE_COLUMNS,
    HealthcareTokenizer,
)

DATASET_PATH = Path("/home/baris/database_transformer/dataset")
ML_PATH = DATASET_PATH / "ml"
DATABASE_PATH = DATASET_PATH / "healthcare.duckdb"
VOCAB_PATH = ML_PATH / "vocab.pt"
EVENT_STAGE_PATH = ML_PATH / "event_stage"
EVENT_STAGE_MARKER = EVENT_STAGE_PATH / "_SUCCESS"

N_BUCKETS = 20

SPLITS = {
    "train": range(0, 16),
    "val": range(16, 18),
    "test": range(18, 20),
}

EVENT_TABLES = [
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

EVENT_TYPE_INDEX_SQL = "CASE o.event_type " + " ".join(
    f"WHEN '{event_type}' THEN {EVENT_TO_INDEX[table]}" for event_type, table in EVENT_TYPES.items()
) + " END::DOUBLE"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def create_integer_id_maps(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE OR REPLACE TEMP TABLE patient_keys AS
        SELECT
            ROW_NUMBER() OVER () - 1 AS patient_key,
            patient_id AS source_patient_id,
            start_time AS birth_time,
            stop_time,
            gender, race, ethnicity, birthplace, city, income
        FROM patients
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE encounter_keys AS
        SELECT
            ROW_NUMBER() OVER () - 1 AS encounter_key,
            encounter_id AS source_encounter_id,
            start_time AS encounter_start_time
        FROM encounters
    """)


def create_events_view(con: duckdb.DuckDBPyConnection) -> None:
    queries = []
    for tbl, event_type, code, val, units, stop_time, extra in EVENT_TABLES:
        where_clause = f"WHERE patient_id IS NOT NULL AND {extra}" if extra else "WHERE patient_id IS NOT NULL"
        source_code = code if code == "NULL" else f"src.{code}"
        source_value = val if val == "NULL" else f"src.{val}"
        source_units = units if units == "NULL" else f"src.{units}"
        source_stop = stop_time if "::" in stop_time else f"src.{stop_time}"
        queries.append(f"""
            SELECT
                patient_keys.patient_key AS patient_id,
                encounter_keys.encounter_key AS encounter_id,
                src.start_time,
                {source_stop} AS stop_time,
                '{event_type}' AS event_type,
                {source_code}::VARCHAR AS code,
                {source_value}::DOUBLE AS value,
                {source_units}::VARCHAR AS units,
                '{tbl}' AS event_table,
                patient_keys.birth_time,
                encounter_keys.encounter_start_time
            FROM {tbl} src
            LEFT JOIN patient_keys ON patient_keys.source_patient_id = src.patient_id
            LEFT JOIN encounter_keys ON encounter_keys.source_encounter_id = src.encounter_id
            {where_clause.replace('patient_id', 'src.patient_id').replace('encounter_id', 'src.encounter_id').replace("type =", "src.type =")}
        """)

    union_sql = " UNION ALL ".join(queries)
    con.execute("DROP VIEW IF EXISTS unified_events")
    con.execute(f"CREATE TEMP VIEW unified_events AS {union_sql}")


def materialize_event_stage(con: duckdb.DuckDBPyConnection) -> None:
    """Scan source Parquets once and cache events partitioned by bucket.

    Without this stage, every bucket re-scans and re-joins all source tables.
    The window functions still run once per bucket, but the expensive source
    scan and patient/encounter mapping happen only once. Delete ``event_stage``
    or set REBUILD_EVENT_STAGE=1 when the source data changes.
    """
    rebuild = os.environ.get("REBUILD_EVENT_STAGE") == "1"
    if EVENT_STAGE_MARKER.exists() and not rebuild:
        log(f"Reusing event stage: {EVENT_STAGE_PATH}")
        return

    if EVENT_STAGE_PATH.exists():
        log(f"Removing event stage: {EVENT_STAGE_PATH}")
        shutil.rmtree(EVENT_STAGE_PATH)

    EVENT_STAGE_PATH.mkdir(parents=True)
    log("Building event stage: scanning source tables once")
    started = time.monotonic()
    con.execute(f"""
        COPY (
            SELECT
                e.*,
                CASE WHEN e.encounter_start_time IS NOT NULL
                          AND e.start_time < e.encounter_start_time
                     THEN e.encounter_start_time ELSE e.start_time END
                    AS normalized_start_time,
                CASE WHEN e.encounter_start_time IS NOT NULL
                          AND e.stop_time < e.encounter_start_time
                     THEN e.encounter_start_time ELSE e.stop_time END
                    AS normalized_stop_time,
                abs(hash(e.patient_id)) % {N_BUCKETS} AS bucket
            FROM unified_events e
        )
        TO '{EVENT_STAGE_PATH.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket),
         ROW_GROUP_SIZE 500000, OVERWRITE_OR_IGNORE);
    """)
    EVENT_STAGE_MARKER.touch()
    log(f"Event stage complete in {time.monotonic() - started:.1f}s")


def create_token_vocab(con, tokenizer):
    vocab_df = pd.DataFrame(tokenizer.db_vocabulary(), columns=["token_key", "token_id"])
    con.register("vocab_df", vocab_df)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE token_vocab AS
        SELECT token_key::VARCHAR AS token_key, token_id::BIGINT AS token_id FROM vocab_df
    """)
    con.unregister("vocab_df")


def materialize_tokenized_events_table(con, tokenizer):
    """Materializes the dataset into a TEMP TABLE once so sorting/windowing 
    and vocab joins are NOT repeated 20 times during export."""
    table_token_sql = "CASE o.event_type " + " ".join(
        f"WHEN '{event_type}' THEN {tokenizer.encode(table)}"
        for event_type, table in EVENT_TYPES.items()
    ) + " END::DOUBLE"
    column_token_sql = "CASE o.event_type " + " ".join(
        f"WHEN '{event_type}' THEN {tokenizer.encode(f'{table}.code')}"
        for event_type, table in EVENT_TYPES.items()
    ) + " END::DOUBLE"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE materialized_events AS
        WITH normalized AS (
            SELECT e.*,
                 CASE WHEN e.encounter_start_time IS NOT NULL
                         AND e.start_time < e.encounter_start_time
                     THEN e.encounter_start_time ELSE e.start_time END AS normalized_start_time,
                 CASE WHEN e.encounter_start_time IS NOT NULL
                         AND e.stop_time < e.encounter_start_time
                     THEN e.encounter_start_time ELSE e.stop_time END AS normalized_stop_time
            FROM unified_events e
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
                ) AS previous_start_time,
                ROW_NUMBER() OVER (
                    PARTITION BY patient_id
                    ORDER BY CAST(normalized_start_time AS DATE) ASC NULLS LAST,
                        CASE WHEN event_type = 'encounter' THEN 0 ELSE 1 END,
                        normalized_start_time ASC NULLS LAST,
                        normalized_stop_time ASC NULLS LAST,
                        encounter_id ASC NULLS LAST,
                        event_type ASC,
                        code ASC
                ) - 1 AS "index"
            FROM normalized n
        )
        SELECT 
            o.* EXCLUDE (normalized_start_time, normalized_stop_time, previous_start_time, event_table, birth_time, encounter_start_time),
            code_v.token_id AS token_0,
            CASE WHEN o.value IS NULL THEN 'nan'::DOUBLE ELSE o.value END AS token_1,
            CASE WHEN o.value IS NULL THEN 0.0 ELSE 1.0 END AS token_2,
            CASE WHEN o.normalized_start_time IS NULL OR o.birth_time IS NULL THEN 'inf'::DOUBLE
                 ELSE epoch(o.normalized_start_time) - epoch(o.birth_time) END AS token_3,
            CASE WHEN o.normalized_stop_time IS NULL OR o.normalized_start_time IS NULL
                      THEN 'inf'::DOUBLE
                  ELSE GREATEST(epoch(o.normalized_stop_time) - epoch(o.normalized_start_time), 0.0)
            END AS token_4,
            CASE WHEN o.previous_start_time IS NULL OR o.normalized_start_time IS NULL
                      THEN 0.0
                  ELSE epoch(o.normalized_start_time) - epoch(o.previous_start_time) END AS token_5,
            {table_token_sql} AS token_6, 
            {column_token_sql} AS token_7,
            {EVENT_TYPE_INDEX_SQL} AS token_8,
            abs(hash(o.patient_id)) % {N_BUCKETS} AS bucket
        FROM ordered o
        LEFT JOIN token_vocab code_v ON code_v.token_key =
            CASE WHEN o.code IS NULL THEN '__NULL__' ELSE o.event_table || '.code.' || o.code END
    """)


def export_events(con: duckdb.DuckDBPyConnection) -> None:
    for split, buckets in SPLITS.items():
        out_dir = ML_PATH / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for bucket in buckets:
            out_path = out_dir / f"part-{bucket:02d}.parquet"
            out_path.unlink(missing_ok=True)

            con.execute(f"""
                COPY (
                    SELECT *
                    FROM materialized_events
                    WHERE bucket = {bucket}
                    ORDER BY patient_id, "index"
                )
                TO '{out_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000);
            """)


def build_patient_index(con: duckdb.DuckDBPyConnection, split: str) -> None:
    split_dir = ML_PATH / split
    index_path = split_dir / "patient_index.parquet"
    index_path.unlink(missing_ok=True)
    log(f"Building {split} patient index")
    started = time.monotonic()

    value_tokens = [f"{column}_v.token_id::DOUBLE" for column in PATIENT_DISCRETE_COLUMNS]
    value_tokens.extend(["null_v.token_id::DOUBLE", "null_v.token_id::DOUBLE"])
    patient_token_0 = f"list_value({', '.join(value_tokens)})"
    patient_token_1 = "list_value(" + ", ".join(
        "CASE WHEN p.income IS NULL THEN 'nan'::DOUBLE ELSE p.income END"
        if column == "income" else "'nan'::DOUBLE"
        for column in PATIENT_FEATURE_COLUMNS
    ) + ")"
    patient_token_2 = "list_value(" + ", ".join(
        "CASE WHEN p.income IS NULL THEN 0.0 ELSE 1.0 END"
        if column == "income" else "0.0"
        for column in PATIENT_FEATURE_COLUMNS
    ) + ")"
    patient_start = "CASE WHEN p.birth_time IS NULL THEN 'inf'::DOUBLE ELSE epoch(p.birth_time) END"
    patient_duration = "CASE WHEN p.stop_time IS NULL OR p.birth_time IS NULL THEN 'inf'::DOUBLE ELSE GREATEST(epoch(p.stop_time) - epoch(p.birth_time), 0.0) END"
    patient_token_3 = f"list_value({', '.join([patient_start] * len(PATIENT_FEATURE_COLUMNS))})"
    patient_token_4 = f"list_value({', '.join([patient_duration] * len(PATIENT_FEATURE_COLUMNS))})"
    patient_token_5 = f"list_value({', '.join(['0.0'] * len(PATIENT_FEATURE_COLUMNS))})"
    patient_token_6 = f"list_value({', '.join(['patients_v.token_id::DOUBLE'] * len(PATIENT_FEATURE_COLUMNS))})"
    patient_token_7 = "list_value(" + ", ".join(
        f"{column}_column_v.token_id::DOUBLE" for column in PATIENT_FEATURE_COLUMNS
    ) + ")"
    patient_token_8 = f"list_value({', '.join(['0.0'] * len(PATIENT_FEATURE_COLUMNS))})"
    patient_value_joins = "\n            ".join(
        f"LEFT JOIN token_vocab {column}_v ON {column}_v.token_key = CASE WHEN p.{column} IS NULL THEN '__NULL__' ELSE 'patients.{column}.' || p.{column} END"
        for column in PATIENT_DISCRETE_COLUMNS
    )
    patient_column_joins = "\n            ".join(
        f"LEFT JOIN token_vocab {column}_column_v ON {column}_column_v.token_key = 'patients.{column}'"
        for column in PATIENT_FEATURE_COLUMNS
    )

    con.execute(f"""
        COPY (
            WITH patient_events AS (
                SELECT
                    patient_id,
                    regexp_extract(filename, 'part-([0-9]+)[.]parquet$', 1)::INTEGER AS bucket,
                    regexp_extract(filename, '(part-[0-9]+[.]parquet)$', 1) AS event_file,
                    MIN(start_time) AS first_event_time,
                    MAX(start_time) AS last_event_time,
                    COUNT(*) AS n_events
                FROM read_parquet('{(split_dir / "part-*.parquet").as_posix()}', filename=true)
                GROUP BY patient_id, bucket, event_file
            )
            SELECT
                e.patient_id,
                p.birth_time AS patient_start_time,
                p.stop_time AS patient_stop_time,
                p.gender, p.race, p.ethnicity, p.birthplace, p.city, p.income,
                e.bucket,
                e.event_file,
                e.first_event_time,
                e.last_event_time,
                e.n_events,
                {patient_token_0} AS patient_token_0,
                {patient_token_1} AS patient_token_1,
                {patient_token_2} AS patient_token_2,
                {patient_token_3} AS patient_token_3,
                {patient_token_4} AS patient_token_4,
                {patient_token_5} AS patient_token_5,
                {patient_token_6} AS patient_token_6,
                {patient_token_7} AS patient_token_7,
                {patient_token_8} AS patient_token_8
            FROM patient_events e
            LEFT JOIN patient_keys p ON p.patient_key = e.patient_id
            {patient_value_joins}
            LEFT JOIN token_vocab null_v ON null_v.token_key = '__NULL__'
            LEFT JOIN token_vocab patients_v ON patients_v.token_key = 'patients'
            {patient_column_joins}
        )
        TO '{index_path.as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000);
    """)
    log(f"Finished {index_path} in {time.monotonic() - started:.1f}s")

def process_and_export_by_bucket(con: duckdb.DuckDBPyConnection, tokenizer) -> None:
    """Processes 1 bucket at a time in RAM (~75M rows each) to prevent 40GB disk spilling."""
    table_token_sql = "CASE o.event_type " + " ".join(
        f"WHEN '{event_type}' THEN {tokenizer.encode(table)}"
        for event_type, table in EVENT_TYPES.items()
    ) + " END::DOUBLE"
    
    column_token_sql = "CASE o.event_type " + " ".join(
        f"WHEN '{event_type}' THEN {tokenizer.encode(f'{table}.code')}"
        for event_type, table in EVENT_TYPES.items()
    ) + " END::DOUBLE"

    for split, buckets in SPLITS.items():
        out_dir = ML_PATH / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for bucket in buckets:
            out_path = out_dir / f"part-{bucket:02d}.parquet"
            out_path.unlink(missing_ok=True)
            started = time.monotonic()
            log(f"Exporting {split} bucket {bucket}/{N_BUCKETS - 1}")

            con.execute(f"""
                COPY (
                    WITH
                    normalized AS (
                        SELECT *
                        FROM read_parquet(
                            '{EVENT_STAGE_PATH.as_posix()}/bucket={bucket}/*.parquet',
                            hive_partitioning=true
                        )
                    ),
                    ordered AS (
                        SELECT n.* EXCLUDE (bucket),
                            LAG(normalized_start_time) OVER (
                                PARTITION BY patient_id
                                ORDER BY CAST(normalized_start_time AS DATE) ASC NULLS LAST,
                                    CASE WHEN event_type = 'encounter' THEN 0 ELSE 1 END,
                                    normalized_start_time ASC NULLS LAST,
                                    normalized_stop_time ASC NULLS LAST,
                                    encounter_id ASC NULLS LAST,
                                    event_type ASC, code ASC
                            ) AS previous_start_time,
                            ROW_NUMBER() OVER (
                                PARTITION BY patient_id
                                ORDER BY CAST(normalized_start_time AS DATE) ASC NULLS LAST,
                                    CASE WHEN event_type = 'encounter' THEN 0 ELSE 1 END,
                                    normalized_start_time ASC NULLS LAST,
                                    normalized_stop_time ASC NULLS LAST,
                                    encounter_id ASC NULLS LAST,
                                    event_type ASC, code ASC
                            ) - 1 AS "index"
                        FROM normalized n
                    )
                    SELECT 
                        o.* EXCLUDE (normalized_start_time, normalized_stop_time, previous_start_time, event_table, birth_time, encounter_start_time),
                        code_v.token_id AS token_0,
                        CASE WHEN o.value IS NULL THEN 'nan'::DOUBLE ELSE o.value END AS token_1,
                        CASE WHEN o.value IS NULL THEN 0.0 ELSE 1.0 END AS token_2,
                        CASE WHEN o.normalized_start_time IS NULL OR o.birth_time IS NULL THEN 'inf'::DOUBLE
                             ELSE epoch(o.normalized_start_time) - epoch(o.birth_time) END AS token_3,
                        CASE WHEN o.normalized_stop_time IS NULL OR o.normalized_start_time IS NULL THEN 'inf'::DOUBLE
                             ELSE GREATEST(epoch(o.normalized_stop_time) - epoch(o.normalized_start_time), 0.0) END AS token_4,
                        CASE WHEN o.previous_start_time IS NULL OR o.normalized_start_time IS NULL THEN 0.0
                             ELSE epoch(o.normalized_start_time) - epoch(o.previous_start_time) END AS token_5,
                        {table_token_sql} AS token_6, 
                        {column_token_sql} AS token_7,
                        {EVENT_TYPE_INDEX_SQL} AS token_8
                    FROM ordered o
                    LEFT JOIN token_vocab code_v ON code_v.token_key =
                        CASE WHEN o.code IS NULL THEN '__NULL__' ELSE o.event_table || '.code.' || o.code END
                    ORDER BY patient_id, "index"
                )
                TO '{out_path.as_posix()}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 500000);
            """)
            log(f"Finished {out_path} in {time.monotonic() - started:.1f}s")

def main() -> None:
    ML_PATH.mkdir(parents=True, exist_ok=True)
    log("Starting healthcare dataset export")
    tokenizer = HealthcareTokenizer(VOCAB_PATH)

    db_config = {
        "threads": "8",
        "memory_limit": "24GB",
        "preserve_insertion_order": "false",
        "temp_directory": (ML_PATH / "duckdb_tmp").as_posix(),
    }

    with duckdb.connect(DATABASE_PATH.as_posix(), config=db_config) as con:
        log("Connected to DuckDB")
        create_token_vocab(con, tokenizer)
        log("Loaded token vocabulary")
        create_integer_id_maps(con)
        log("Created patient and encounter ID maps")
        create_events_view(con)
        log("Created unified events view")

        # 1. Scan source tables once; subsequent exports read one small stage
        # partition per bucket instead of re-scanning every source table.
        materialize_event_stage(con)

        # 2. Process & export Parquet files one bucket at a time
        process_and_export_by_bucket(con, tokenizer)

        # 3. Build indexes from exported Parquets
        for split in SPLITS:
            build_patient_index(con, split)

    log("Healthcare dataset export complete")




if __name__ == "__main__":
    main()