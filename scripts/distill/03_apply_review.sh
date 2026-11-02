#!/usr/bin/env bash
# Step 3: apply the completed review CSV back to the data.
#
# This is NOT another review pass. The human review happens by editing
# data/distill/filtered/<run_id>.review.csv:
#   1. model_verdict / model_notes  (first pass)
#   2. human_verdict / human_notes  (second pass; overrides the model)
# Empty cells keep the automatic verdict. This script only writes those
# decisions back: rejects are removed, and rescued rows are restored.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" -m distillation.run_review merge \
    --filtered "$FILTERED" \
    --rejected "$REJECTED" \
    --review "$REVIEW" \
    --out "$REVIEWED" \
    --rejected-out "$REVIEWED_REJECTED" \
    --stats-out "$REVIEW_STATS" \
    "$@"
