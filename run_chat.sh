#!/usr/bin/env bash
# Launch quin-llm chat mode with a checkpoint.
#
# Usage:
#   ./run_chat.sh [options]
#   ./run_chat.sh --mode <full|mini> [options]
#
# mode:
#   full   use checkpoints/sft_full/best.pt (default)
#   mini   use checkpoints/sft_mini/best.pt
#
# Extra options are passed to `python main.py chat`, e.g. --temperature 0.7.
# Set CKPT to override the checkpoint path.
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./run_chat.sh [options]

  --mode full|mini   choose the default checkpoint (default: full)
  --ckpt PATH        checkpoint path (same as main.py chat --ckpt)
  -h, --help         show this help

Any other option is forwarded to `python main.py chat`.
The CKPT environment variable can also override the default checkpoint.

Examples:
  ./run_chat.sh
  ./run_chat.sh --mode mini
  CKPT=checkpoints/sft_full/final.pt ./run_chat.sh
  ./run_chat.sh --temperature 0.7 --max-new-tokens 300
EOF
}

MODE="${MODE:-full}"
PASSTHROUGH=()

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --mode)
            if (($# < 2)); then
                echo "error: --mode requires a value (full or mini)" >&2
                exit 2
            fi
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --)
            shift
            PASSTHROUGH+=("$@")
            break
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

MODE="${MODE,,}"
case "$MODE" in
    full)
        DEFAULT_CKPT="checkpoints/sft_full/best.pt"
        ;;
    mini)
        DEFAULT_CKPT="checkpoints/sft_mini/best.pt"
        ;;
    *)
        echo "error: invalid mode '$MODE' (expected full or mini)" >&2
        exit 2
        ;;
esac

HAS_CKPT=0
if ((${#PASSTHROUGH[@]})); then
    for arg in "${PASSTHROUGH[@]}"; do
        if [[ "$arg" == "--ckpt" || "$arg" == --ckpt=* ]]; then
            HAS_CKPT=1
        fi
    done
fi

CKPT_ARGS=()
if ((HAS_CKPT == 0)); then
    CKPT="${CKPT:-$DEFAULT_CKPT}"
    if [[ ! -f "$CKPT" ]]; then
        echo "error: checkpoint not found: $CKPT" >&2
        echo "hint: set CKPT=... or pass --ckpt PATH; for the CPU demo use --mode mini" >&2
        exit 1
    fi
    CKPT_ARGS=(--ckpt "$CKPT")
    echo "chat: mode=$MODE checkpoint=$CKPT"
else
    echo "chat: checkpoint=<from --ckpt>"
fi

PYTHON="${PYTHON:-python}"
CMD=("$PYTHON" main.py chat)
if ((${#CKPT_ARGS[@]})); then
    CMD+=("${CKPT_ARGS[@]}")
fi
if ((${#PASSTHROUGH[@]})); then
    CMD+=("${PASSTHROUGH[@]}")
fi
exec "${CMD[@]}"
