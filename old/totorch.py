from pathlib import Path
import duckdb

DATASET_PATH = Path("/home/baris/database_transformer/dataset")
ML_PATH = DATASET_PATH / "ml"
DATABASE_PATH = DATASET_PATH / "healthcare.duckdb"

def export_grouped_sequences():
    ML_PATH.mkdir(parents=True, exist_ok=True)

    with duckdb.connect(str(DATABASE_PATH)) as con:
        con.execute("SET memory_limit='50GB'")
        # LIST aggregation is memory intensive.  Disable DuckDB's result-order
        # buffer and leave enough memory for the OS/filesystem cache.
        con.execute("SET threads=24")
        con.execute("SET preserve_insertion_order=false")
        con.execute(f"SET temp_directory='{ML_PATH / 'duckdb_tmp'}'")

        print("Exporting train/val/test splits to partitioned Parquet...")

        # Base CTE for all events
        base_query = """
            WITH events AS (
                SELECT patient_id, encounter_id, start_time, stop_time, 'encounter' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM encounters WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, stop_time, 'condition' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM conditions WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, stop_time, 'medication' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM medications WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, stop_time, 'procedure' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM procedures WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, NULL::TIMESTAMP AS stop_time, 'observation_numeric' AS event_type, code::VARCHAR AS code, value::VARCHAR AS value, units::VARCHAR AS units FROM observations WHERE patient_id IS NOT NULL AND type='numeric'
                UNION ALL
                SELECT patient_id, encounter_id, start_time, NULL::TIMESTAMP AS stop_time, 'immunization' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM immunizations WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, stop_time, 'allergy' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM allergies WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, stop_time, 'careplan' AS event_type, code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM careplans WHERE patient_id IS NOT NULL
                UNION ALL
                SELECT patient_id, encounter_id, start_time, NULL::TIMESTAMP AS stop_time, 'imaging' AS event_type, bodysite_code::VARCHAR AS code, NULL::VARCHAR AS value, NULL::VARCHAR AS units FROM imaging_studies WHERE patient_id IS NOT NULL
            ),
            patient_events AS (
                SELECT 
                    p.patient_id,
                    p.start_time AS patient_start_time,
                    p.stop_time AS patient_stop_time,
                    MIN(e.start_time) AS first_event_time,
                    MAX(COALESCE(e.stop_time, e.start_time)) AS last_event_time,
                    p.gender AS patient_gender,
                    p.race AS patient_race,
                    p.ethnicity AS patient_ethnicity,
                    p.birthplace AS patient_birthplace,
                    p.city AS patient_city,
                    p.income AS patient_income,
                    LIST(
                        STRUCT_PACK(
                            encounter_id := e.encounter_id,
                            start_time := e.start_time,
                            stop_time := e.stop_time,
                            event_type := e.event_type,
                            code := e.code,
                            value := e.value,
                            units := e.units
                        )
                        ORDER BY
                            e.encounter_id ASC NULLS LAST,
                            CASE WHEN e.event_type = 'encounter' THEN 0 ELSE 1 END,
                            e.start_time ASC NULLS LAST,
                            e.stop_time ASC NULLS LAST,
                            e.event_type ASC,
                            e.code ASC
                    ) AS events
                FROM patients p
                INNER JOIN events e ON p.patient_id = e.patient_id
                WHERE abs(hash(p.patient_id)) % 20 = {{bucket}}
                GROUP BY p.patient_id, p.start_time, p.stop_time, p.gender, p.race, p.ethnicity, p.birthplace, p.city, p.income
            )
        """

        # A single Parquet file requires DuckDB to hold every patient's LIST
        # aggregate at once.  Write one file per hash bucket instead.  A
        # directory of Parquet files is a standard Arrow/DuckDB dataset and
        # can be passed to read_table/read_parquet as a dataset path.
        splits = {
            "train_parts": range(0, 16),
            "val_parts": range(16, 18),
            "test_parts": range(18, 20),
        }

        for directory, buckets in splits.items():
            out_dir = ML_PATH / directory
            out_dir.mkdir(parents=True, exist_ok=True)

            for bucket in buckets:
                out_path = out_dir / f"part-{bucket:02d}.parquet"
                query = base_query.replace("{{bucket}}", str(bucket))
                print(f"  {directory}/{out_path.name}")
                # Overwrite an interrupted previous run's part.
                out_path.unlink(missing_ok=True)
                con.execute(f"""
                    COPY (
                        {query}
                        SELECT * FROM patient_events
                    )
                    TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)

# Instantiating in PyTorch
#train_ds = PatientSequenceDataset(ML_PATH / "train.parquet")
#val_ds = PatientSequenceDataset(ML_PATH / "val.parquet")
#test_ds = PatientSequenceDataset(ML_PATH / "test.parquet")

export_grouped_sequences()