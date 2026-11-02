#!/usr/bin/env bash
# Step 6: generate controlled Baseline and Mixed SFT configs with the same
# max_steps, seed, replay, lr, and warmup. Early stopping is disabled.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" -m distillation.make_configs \
    --run-id "$RUN_ID" \
    --mixed-data "${DATASET_DIR}/train.parquet" \
    "$@"
