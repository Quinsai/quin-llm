#!/usr/bin/env bash
# Step 0: check environment, data, checkpoints, teacher API, and baseline budget.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

mkdir -p "$REPORT_DIR"
"$PYTHON" -m distillation.preflight \
    --teacher-config "$TEACHER_CONFIG" \
    --json-out "${REPORT_DIR}/preflight_${RUN_ID}.json" \
    "$@"
