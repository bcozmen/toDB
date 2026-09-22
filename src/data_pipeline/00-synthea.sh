#!/bin/bash

PATIENT_COUNT="${1:-2000000}"
SEED="${2:-12345}"

echo "Running Synthea with:"
echo "  Patients: $PATIENT_COUNT"
echo "  Seed:     $SEED"

nohup ./synthea/run_synthea \
    -p "$PATIENT_COUNT" \
    -s "$SEED" \
    --exporter.csv.export=true \
    --exporter.fhir.export=false \
    --exporter.csv.excluded_files=patient_expenses.csv,payers.csv,providers.csv,organizations.csv,payer_transitions.csv,claims_transactions.csv,claims.csv \
    > synthea.log 2>&1 &
