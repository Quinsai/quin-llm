#!/usr/bin/env bash
# Run quin-llm stages from PHASE through sft.
#
# Usage:
#   ./run_pipeline.sh [phase] [mode]
#   ./run_pipeline.sh --mode <full|mini> [phase]
#
# phase: tokenize | prepare-pretrain | pretrain | prepare-sft | sft
#        default: pretrain
# mode:  full (default, GPU configs) | mini (CPU smoke configs)
#
# Examples:
#   ./run_pipeline.sh tokenize
#   ./run_pipeline.sh prepare-sft mini
#   ./run_pipeline.sh --mode mini tokenize
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: ./run_pipeline.sh [phase] [mode]

phase (default: pretrain):
  tokenize           train the BPE tokenizer
  prepare-pretrain   build pretrain token files
  pretrain           train the base model
  prepare-sft        build token-level SFT samples
  sft                fine-tune on instructions

mode (default: full):
  full               GPU/full configs
  mini               CPU-only smoke configs

The script runs the selected phase and every later phase through sft.

Examples:
  ./run_pipeline.sh tokenize
  ./run_pipeline.sh prepare-sft mini
  ./run_pipeline.sh --mode mini tokenize
EOF
}

PHASE="pretrain"
MODE="full"
POSITIONAL=()

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
            POSITIONAL+=("$@")
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if ((${#POSITIONAL[@]} > 2)); then
    echo "error: too many arguments" >&2
    usage >&2
    exit 2
fi
if ((${#POSITIONAL[@]} >= 1)); then
    PHASE="${POSITIONAL[0]}"
fi
if ((${#POSITIONAL[@]} >= 2)); then
    MODE="${POSITIONAL[1]}"
fi

PHASE="${PHASE//_/-}"
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
for i in "${!PHASES[@]}"; do
    if [[ "${PHASES[$i]}" == "$PHASE" ]]; then
        START_INDEX=$i
        break
    fi
done

if ((START_INDEX < 0)); then
    echo "error: invalid phase '$PHASE'" >&2
    echo "expected one of: ${PHASES[*]}" >&2
    exit 2
fi

require_pretrain_ckpt() {
    if [[ ! -f "$PRETRAIN_CKPT" ]]; then
        echo "error: pretrain checkpoint not found: $PRETRAIN_CKPT" >&2
        echo "hint: run './run_pipeline.sh pretrain $MODE' first, or set PRETRAIN_CKPT" >&2
        exit 1
    fi
}

run_phase() {
    local phase="$1"
    echo
    echo "==> $phase"
    case "$phase" in
        tokenize)
            "$PYTHON" main.py tokenize --config "$TOKENIZER_CONFIG"
            ;;
        prepare-pretrain)
            "$PYTHON" main.py prepare-pretrain --config "$PRETRAIN_CONFIG"
            ;;
        pretrain)
            "$PYTHON" main.py pretrain --config "$PRETRAIN_CONFIG"
            ;;
        prepare-sft)
            require_pretrain_ckpt
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

echo "pipeline: phase=$PHASE mode=$MODE"
echo "config: tokenizer=$TOKENIZER_CONFIG"
echo "config: pretrain=$PRETRAIN_CONFIG"
echo "config: sft=$SFT_CONFIG"
echo "ckpt:   pretrain=$PRETRAIN_CKPT"
echo "python: $PYTHON"

for ((i = START_INDEX; i < ${#PHASES[@]}; i++)); do
    run_phase "${PHASES[$i]}"
done

echo
echo "pipeline finished: phase=$PHASE mode=$MODE"
