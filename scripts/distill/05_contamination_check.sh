#!/usr/bin/env bash
# Step 5: check the final SFT parquet (teacher + Alpaca) against the frozen
# evaluation sets. A non-zero exit means do not train until fixed.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

mkdir -p "$REPORT_DIR"
"$PYTHON" -m evaluation.contamination_check \
    --train "${DATASET_DIR}/train.parquet" \
    --eval evaluation/datasets \
    --json-out "${REPORT_DIR}/contamination_${RUN_ID}.json" \
    "$@"
