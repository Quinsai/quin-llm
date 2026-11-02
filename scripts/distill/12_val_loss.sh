#!/usr/bin/env bash
# Step 12: pretraining validation loss/perplexity for both checkpoints, as a
# retention / catastrophic-forgetting check.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

mkdir -p "$REPORT_DIR"
"$PYTHON" -m evaluation.val_loss \
    --ckpt "$BASELINE_CKPT" \
    --data-dir processed_data/pretrain \
    --json-out "${REPORT_DIR}/val_loss_baseline_${RUN_ID}.json" \
    "$@"
"$PYTHON" -m evaluation.val_loss \
    --ckpt "$MIXED_CKPT" \
    --data-dir processed_data/pretrain \
    --json-out "${REPORT_DIR}/val_loss_mixed_${RUN_ID}.json" \
    "$@"
