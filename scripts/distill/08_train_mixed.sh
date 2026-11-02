#!/usr/bin/env bash
# Step 8: prepare SFT samples and train the Mixed (Alpaca + teacher) model.
set -euo pipefail
source "$(dirname "$0")/_env.sh"

"$PYTHON" main.py prepare-sft \
    --config "$MIXED_CONFIG" \
    --ckpt "$PRETRAIN_CKPT"
"$PYTHON" main.py sft \
    --config "$MIXED_CONFIG" \
    --ckpt "$PRETRAIN_CKPT"
