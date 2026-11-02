#!/usr/bin/env bash
# Step 10: greedy-decode the frozen evaluation sets with the Mixed checkpoint.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

mkdir -p "$REPORT_DIR"
"$PYTHON" -m evaluation.run_eval \
    --config configs/eval.yaml \
    --ckpt "$MIXED_CKPT" \
    --out "$MIXED_EVAL" \
    "$@"
