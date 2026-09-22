Channel | Stream Name | Type | Description
token_0 | Concept Code | BIGINT/DOUBLE | Vocabulary ID for discrete concepts
token_1 | Continuous Value | DOUBLE | Associated numeric value (or NaN)
token_2 | Value Mask | DOUBLE | 1.0 if token_1 exists, 0.0 if NaN
token_3 | Age / Start Epoch | DOUBLE | Elapsed seconds from birth/start (or inf)
token_4 | Duration | DOUBLE | Event/Lifetime duration in seconds
token_5 | Time Delta (dt) | DOUBLE | Seconds since previous event (0.0 for patient)
token_6 | Table ID | BIGINT/DOUBLE | Source table vocabulary ID
token_7 | Column ID | BIGINT/DOUBLE | Source column vocabulary ID
token_8 | Event Type Index | DOUBLE | Mapped index from EVENT_TO_INDEX

Slot | Column | token_0 | token_1 | token_2 | token_3 | token_4 | token_5 | token_6 | token_7 | token_8
0 | gender | gender_v ID | NaN | 0.0 | patient_start | patient_duration | 0.0 | patients_v ID | gender_col ID | 0.0
1 | race | race_v ID | NaN | 0.0 | patient_start | patient_duration | 0.0 | patients_v ID | race_col ID | 0.0
2 | ethnicity | ethnicity_v ID | NaN | 0.0 | patient_start | patient_duration | 0.0 | patients_v ID | ethnicity_col ID | 0.0
3 | birthplace | birthplace_v ID | NaN | 0.0 | patient_start | patient_duration | 0.0 | patients_v ID | birthplace_col ID | 0.0
4 | city | city_v ID | NaN | 0.0 | patient_start | patient_duration | 0.0 | patients_v ID | city_col ID | 0.0
5 | time | null_v ID | NaN | 0.0 | patient_start | patient_duration | 0.0 | patients_v ID | time_col ID | 0.0
6 | income | null_v ID | income | 1.0/0.0 | patient_start | patient_duration | 0.0 | patients_v ID | income_col ID | 0.0

Patient feature order is therefore:

```text
0: gender
1: race
2: ethnicity
3: birthplace
4: city
5: time
6: income
```

The serialized exporter currently repeats patient time fields across all seven
patient slots. `PatientEventsDataset` removes those repeated values at load
time: only slot 5 retains time information. In the runtime model input, slot 5
uses relative patient age (`token_3 = 0`, `token_4 = current age`,
`token_5 = 0`), while the other patient slots use `inf` for their time fields
to represent missing time values.