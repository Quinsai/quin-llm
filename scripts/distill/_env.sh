#!/usr/bin/env bash
# Shared paths for scripts/distill/*.sh.
#
# Experiment settings live in configs/*.yaml.  The run id, teacher config, and
# generation output directories are read from configs/distill_generation.yaml,
# so changing the run only requires editing that YAML.  PYTHON may still be
# overridden to point at the quin-llm interpreter.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
GENERATION_CONFIG="configs/distill_generation.yaml"
FILTER_CONFIG="configs/distill_filter.yaml"
DATASET_CONFIG="configs/distill_dataset.yaml"
EVAL_CONFIG="configs/eval.yaml"
REPORT_DIR="reports"
PRETRAIN_CKPT="checkpoints/pretrain_full/best.pt"

_gen_meta="$(
    "$PYTHON" - "$GENERATION_CONFIG" <<'PY'
import sys

from config import DistillGenerationConfig

cfg = DistillGenerationConfig.from_yaml(sys.argv[1])
print("\t".join([cfg.run_id, cfg.teacher_config, cfg.out_dir, cfg.tasks_dir]))
PY
)" || {
    echo "error: failed to read $GENERATION_CONFIG" >&2
    return 1
}

IFS=$'\t' read -r RUN_ID TEACHER_CONFIG RAW_DIR TASKS_DIR <<< "$_gen_meta"

if [[ -z "${RUN_ID:-}" ]]; then
    echo "error: run_id is empty in $GENERATION_CONFIG" >&2
    return 1
fi

FILTERED_DIR="data/distill/filtered"
DATASET_DIR="data/distill/sft/${RUN_ID}"

RAW="${RAW_DIR}/${RUN_ID}.jsonl"
FILTERED="${FILTERED_DIR}/${RUN_ID}.jsonl"
REJECTED="${FILTERED_DIR}/${RUN_ID}.rejected.jsonl"
STATS="${FILTERED_DIR}/${RUN_ID}.stats.json"
REVIEW="${FILTERED_DIR}/${RUN_ID}.review.csv"
REVIEWED="${FILTERED_DIR}/${RUN_ID}.reviewed.jsonl"
REVIEWED_REJECTED="${FILTERED_DIR}/${RUN_ID}.reviewed.rejected.jsonl"
REVIEW_STATS="${FILTERED_DIR}/${RUN_ID}.review.stats.json"

BASELINE_CONFIG="configs/generated/sft_baseline_${RUN_ID}.yaml"
MIXED_CONFIG="configs/generated/sft_distill_${RUN_ID}.yaml"
BASELINE_CKPT="checkpoints/sft_baseline_${RUN_ID}/best.pt"
MIXED_CKPT="checkpoints/sft_distill_${RUN_ID}/best.pt"
BASELINE_EVAL="${REPORT_DIR}/eval_baseline_${RUN_ID}.jsonl"
MIXED_EVAL="${REPORT_DIR}/eval_mixed_${RUN_ID}.jsonl"
REPORT="${REPORT_DIR}/distill_${RUN_ID}.md"
