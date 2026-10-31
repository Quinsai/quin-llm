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

  --mode full|mini       choose the default checkpoint (default: full)
  --ckpt PATH            checkpoint path (same as main.py chat --ckpt)
  --save-log true|false  save output to log/chat_*.log (default: true)
  -h, --help             show this help

Any other option is forwarded to `python main.py chat`.
The CKPT environment variable can also override the default checkpoint.

Examples:
  ./run_chat.sh
  ./run_chat.sh --mode mini
  ./run_chat.sh --save-log false
  CKPT=checkpoints/sft_full/final.pt ./run_chat.sh
  ./run_chat.sh --temperature 0.7 --max-new-tokens 300
EOF
}

set_save_log() {
    case "${1,,}" in
        true|1|yes|on) SAVE_LOG=1 ;;
        false|0|no|off) SAVE_LOG=0 ;;
        *)
            echo "error: invalid --save-log value '$1' (expected true or false)" >&2
            exit 2
            ;;
    esac
}

MODE="${MODE:-full}"
SAVE_LOG=1
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
        --save-log)
            if (($# < 2)); then
                echo "error: --save-log requires a value (true or false)" >&2
                exit 2
            fi
            set_save_log "$2"
            shift 2
            ;;
        --save-log=*)
            set_save_log "${1#*=}"
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

LOG_DIR="${LOG_DIR:-log}"
if ((SAVE_LOG)); then
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/chat_${MODE}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee "$LOG_FILE") 2>&1
    export PYTHONUNBUFFERED=1
    echo "log: $LOG_FILE"
else
    echo "log: disabled"
fi

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
