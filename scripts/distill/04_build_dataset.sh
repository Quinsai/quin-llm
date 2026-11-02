#!/usr/bin/env bash
# Step 4: mix reviewed teacher data with Alpaca by answer-token ratio (50:50 by
# default) and write data/distill/sft/<run_id>/train.parquet + manifest.json.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" -m distillation.run_build_dataset \
    --config configs/distill_dataset.yaml \
    --filtered "$REVIEWED" \
    --out "$DATASET_DIR" \
    --baseline-budget data/distill/baseline_budget.json \
    "$@"
