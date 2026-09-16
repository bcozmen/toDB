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

        print("Exporting hierarchical train/val/test splits to Parquet...")

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
            -- Level 1: Aggregate events into encounters
            encounter_events AS (
                SELECT 
                    patient_id,
                    COALESCE(encounter_id, 'UNASSIGNED') AS encounter_id,
                    MIN(start_time) AS encounter_start_time,
                    MAX(stop_time) AS encounter_stop_time,
                    LIST(
                        STRUCT_PACK(
                            event_type := event_type,
                            code := code,
                            value := value,
                            units := units,
                            start_time := start_time,
                            stop_time := stop_time
                        )
                        ORDER BY start_time ASC NULLS LAST, stop_time ASC NULLS LAST, event_type ASC, code ASC
                    ) AS events
                FROM events
                WHERE abs(hash(patient_id)) % 50 = {{bucket}}
                GROUP BY patient_id, COALESCE(encounter_id, 'UNASSIGNED')
            ),
            -- Level 2: Aggregate encounters into patients
            patient_events AS (
                SELECT 
                    p.patient_id,
                    p.start_time AS patient_start_time,
                    p.stop_time AS patient_stop_time,
                    p.gender AS patient_gender,
                    p.race AS patient_race,
                    p.ethnicity AS patient_ethnicity,
                    p.birthplace AS patient_birthplace,
                    p.city AS patient_city,
                    p.income AS patient_income,
                    LIST(
                        STRUCT_PACK(
                            encounter_id := ee.encounter_id,
                            start_time := ee.encounter_start_time,
                            stop_time := ee.encounter_stop_time,
                            events := ee.events
                        )
                        ORDER BY ee.encounter_start_time ASC NULLS LAST
                    ) AS encounters
                FROM patients p
                INNER JOIN encounter_events ee ON p.patient_id = ee.patient_id
                GROUP BY p.patient_id, p.start_time, p.stop_time, p.gender, p.race, p.ethnicity, p.birthplace, p.city, p.income
            )
        """

        # Aggregate one patient bucket at a time.  The bucket predicate is
        # applied before encounter LISTs are built, which prevents DuckDB from
        # materializing the complete 1B+ event dataset in one query.
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
                out_path.unlink(missing_ok=True)
                query = base_query.replace("{{bucket}}", str(bucket))
                print(f"  {directory}/{out_path.name}")
                con.execute(f"""
                    COPY (
                        {query}
                        SELECT * FROM patient_events
                    )
                    TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);
                """)

export_grouped_sequences()