#!/usr/bin/env bash
# Run quin-llm stages in the inclusive range [start, end].
#
# Usage:
#   ./run_pipeline.sh [--start STAGE] [--end STAGE] [--mode full|mini] [--save-log true|false]
#
# start default: pretrain
# end default:   sft
# mode default:  full
# save-log default: true
#
# Stage order:
#   tokenize | prepare-pretrain | pretrain | prepare-sft | sft
#
# Examples:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --start tokenize
#   ./run_pipeline.sh --start prepare-pretrain --end pretrain
#   ./run_pipeline.sh --start tokenize --end pretrain --mode mini
#   ./run_pipeline.sh --save-log false
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./run_pipeline.sh [--start STAGE] [--end STAGE] [--mode full|mini] [--save-log true|false]

Data (full mode only):
  The full configs read local parquet files under data/raw/. Stages that need
  them run ./scripts/download_data.sh (ModelScope, resumable) automatically
  when data/raw/.complete is missing.

start (default: pretrain):
  tokenize           train the BPE tokenizer
  prepare-pretrain   build pretrain token files
  pretrain           train the base model
  prepare-sft        build token-level SFT samples
  sft                fine-tune on instructions

end (default: sft):
  same stage names as start

mode (default: full):
  full               GPU/full configs
  mini               CPU-only smoke configs

save-log (default: true):
  true               write stdout/stderr to log/pipeline_*.log
  false              do not write a log file

The script runs every stage from start through end, inclusive.

Examples:
  ./run_pipeline.sh
  ./run_pipeline.sh --start tokenize
  ./run_pipeline.sh --start prepare-pretrain --end pretrain
  ./run_pipeline.sh --start tokenize --end pretrain --mode mini
  ./run_pipeline.sh --save-log false
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

START_PHASE="pretrain"
END_PHASE="sft"
MODE="full"
SAVE_LOG=1

while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --start)
            if (($# < 2)); then
                echo "error: --start requires a value" >&2
                exit 2
            fi
            START_PHASE="$2"
            shift 2
            ;;
        --start=*)
            START_PHASE="${1#*=}"
            shift
            ;;
        --end)
            if (($# < 2)); then
                echo "error: --end requires a value" >&2
                exit 2
            fi
            END_PHASE="$2"
            shift 2
            ;;
        --end=*)
            END_PHASE="${1#*=}"
            shift
            ;;
        --mode)
            if (($# < 2)); then
                echo "error: --mode requires a value" >&2
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
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            echo "error: unexpected positional argument: $1" >&2
            echo "hint: use --start STAGE / --end STAGE / --mode full|mini" >&2
            usage >&2
            exit 2
            ;;
    esac
done

START_PHASE="${START_PHASE//_/-}"
END_PHASE="${END_PHASE//_/-}"
MODE="${MODE,,}"

case "$MODE" in
    full)
        DEFAULT_TOKENIZER_CONFIG="configs/tokenizer.yaml"
        DEFAULT_PRETRAIN_CONFIG="configs/pretrain_full.yaml"
        DEFAULT_SFT_CONFIG="configs/sft_full.yaml"
        DEFAULT_PRETRAIN_CKPT="checkpoints/pretrain_full/best.pt"
        ;;
    mini)
        DEFAULT_TOKENIZER_CONFIG="configs/tokenizer_mini.yaml"
        DEFAULT_PRETRAIN_CONFIG="configs/pretrain_mini.yaml"
        DEFAULT_SFT_CONFIG="configs/sft_mini.yaml"
        DEFAULT_PRETRAIN_CKPT="checkpoints/pretrain_mini/best.pt"
        ;;
    *)
        echo "error: invalid mode '$MODE' (expected full or mini)" >&2
        exit 2
        ;;
esac

TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-$DEFAULT_TOKENIZER_CONFIG}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-$DEFAULT_PRETRAIN_CONFIG}"
SFT_CONFIG="${SFT_CONFIG:-$DEFAULT_SFT_CONFIG}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-$DEFAULT_PRETRAIN_CKPT}"
PYTHON="${PYTHON:-python}"

PHASES=(tokenize prepare-pretrain pretrain prepare-sft sft)
START_INDEX=-1
END_INDEX=-1
for i in "${!PHASES[@]}"; do
    if [[ "${PHASES[$i]}" == "$START_PHASE" ]]; then
        START_INDEX=$i
    fi
    if [[ "${PHASES[$i]}" == "$END_PHASE" ]]; then
        END_INDEX=$i
    fi
done

if ((START_INDEX < 0)); then
    echo "error: invalid start stage '$START_PHASE'" >&2
    echo "expected one of: ${PHASES[*]}" >&2
    exit 2
fi
if ((END_INDEX < 0)); then
    echo "error: invalid end stage '$END_PHASE'" >&2
    echo "expected one of: ${PHASES[*]}" >&2
    exit 2
fi
if ((START_INDEX > END_INDEX)); then
    echo "error: start stage '$START_PHASE' comes after end stage '$END_PHASE'" >&2
    exit 2
fi

LOG_DIR="${LOG_DIR:-log}"
if ((SAVE_LOG)); then
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/pipeline_${MODE}_$(date +%Y%m%d_%H%M%S).log"
    exec > >(tee "$LOG_FILE") 2>&1
    export PYTHONUNBUFFERED=1
    echo "log: $LOG_FILE"
else
    echo "log: disabled"
fi

require_pretrain_ckpt() {
    if [[ ! -f "$PRETRAIN_CKPT" ]]; then
        echo "error: pretrain checkpoint not found: $PRETRAIN_CKPT" >&2
        echo "hint: run './run_pipeline.sh --start pretrain --end pretrain --mode $MODE' first, or set PRETRAIN_CKPT" >&2
        exit 1
    fi
}

require_data() {
    # mini mode uses pipeline/fixtures, not data/raw.
    if [[ "$MODE" == "mini" || -f data/raw/.complete ]]; then
        return
    fi
    echo "==> data/raw incomplete, running scripts/download_data.sh"
    ./scripts/download_data.sh
}

run_phase() {
    local phase="$1"
    echo
    echo "==> $phase"
    case "$phase" in
        tokenize)
            require_data
            "$PYTHON" main.py tokenize --config "$TOKENIZER_CONFIG"
            ;;
        prepare-pretrain)
            require_data
            "$PYTHON" main.py prepare-pretrain --config "$PRETRAIN_CONFIG"
            ;;
        pretrain)
            "$PYTHON" main.py pretrain --config "$PRETRAIN_CONFIG"
            ;;
        prepare-sft)
            require_pretrain_ckpt
            require_data
            "$PYTHON" main.py prepare-sft --config "$SFT_CONFIG" --ckpt "$PRETRAIN_CKPT"
            ;;
        sft)
            require_pretrain_ckpt
            "$PYTHON" main.py sft --config "$SFT_CONFIG" --ckpt "$PRETRAIN_CKPT"
            ;;
        *)
            echo "error: internal error: unknown phase '$phase'" >&2
            exit 1
            ;;
    esac
}

echo "pipeline: start=$START_PHASE end=$END_PHASE mode=$MODE"
echo "config: tokenizer=$TOKENIZER_CONFIG"
echo "config: pretrain=$PRETRAIN_CONFIG"
echo "config: sft=$SFT_CONFIG"
echo "ckpt:   pretrain=$PRETRAIN_CKPT"
echo "python: $PYTHON"

for ((i = START_INDEX; i <= END_INDEX; i++)); do
    run_phase "${PHASES[$i]}"
done

echo
echo "pipeline finished: start=$START_PHASE end=$END_PHASE mode=$MODE"
