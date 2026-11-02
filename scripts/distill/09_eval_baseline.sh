#!/usr/bin/env bash
# Step 9: greedy-decode the frozen evaluation sets with the Baseline checkpoint.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

mkdir -p "$REPORT_DIR"
"$PYTHON" -m evaluation.run_eval \
    --config configs/eval.yaml \
    --ckpt "$BASELINE_CKPT" \
    --out "$BASELINE_EVAL" \
    "$@"
