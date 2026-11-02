#!/usr/bin/env bash
# Step 1: generate teacher data from configs/distill_generation.yaml.
#
# Edit that YAML to change the run id, category counts, or output directories.
# Extra arguments are passed to distillation.run_generation, for example:
#   ./scripts/distill/01_generate.sh --limit 20
#   ./scripts/distill/01_generate.sh --dry-run --limit 5
set -euo pipefail
source "$(dirname "$0")/_env.sh"

echo "==> run_id: $RUN_ID"
echo "==> config: $GENERATION_CONFIG"

"$PYTHON" -m distillation.run_generation \
    --config "$GENERATION_CONFIG" \
    --teacher-config "$TEACHER_CONFIG" \
    "$@"
