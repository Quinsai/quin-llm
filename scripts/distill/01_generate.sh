#!/usr/bin/env bash
# Step 1: generate teacher data. Pilot by default; edit the category counts in
# configs/distill_generation.yaml for the ~40k main run.
#
# Examples:
#   ./scripts/distill/01_generate.sh --dry-run --limit 5
#   ./scripts/distill/01_generate.sh --limit 5
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" -m distillation.run_generation \
    --config configs/distill_generation.yaml \
    --teacher-config "$TEACHER_CONFIG" \
    --run-id "$RUN_ID" \
    "$@"
