#!/usr/bin/env bash
# Step 7: prepare SFT samples and train the Alpaca-only controlled Baseline.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" main.py prepare-sft \
    --config "$BASELINE_CONFIG" \
    --ckpt "$PRETRAIN_CKPT"
"$PYTHON" main.py sft \
    --config "$BASELINE_CONFIG" \
    --ckpt "$PRETRAIN_CKPT"
