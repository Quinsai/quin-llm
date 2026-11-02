#!/usr/bin/env bash
# Step 2: filter raw teacher output, deduplicate, check eval contamination, and
# write the stratified review CSV used for the two-pass manual review.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" -m distillation.run_filter \
    --raw "$RAW" \
    --out "$FILTERED" \
    --config "$FILTER_CONFIG" \
    --rejected-out "$REJECTED" \
    --stats-out "$STATS" \
    --review-out "$REVIEW" \
    "$@"
