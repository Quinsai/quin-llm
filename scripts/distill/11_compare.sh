#!/usr/bin/env bash
# Step 11: paired Baseline vs Mixed comparison with bootstrap CI and McNemar.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

mkdir -p "$REPORT_DIR"
"$PYTHON" -m evaluation.compare_models \
    --baseline "$BASELINE_EVAL" \
    --candidate "$MIXED_EVAL" \
    --report "$REPORT" \
    "$@"
