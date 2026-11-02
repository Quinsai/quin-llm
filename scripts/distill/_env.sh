#!/usr/bin/env bash
# Shared defaults for scripts/distill/*.sh. Override with environment variables:
#   RUN_ID, PYTHON, TEACHER_CONFIG, REPORT_DIR, PRETRAIN_CKPT
#
# Each numbered script sources this file and then runs one pipeline step.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
RUN_ID="${RUN_ID:-pilot_v1}"
TEACHER_CONFIG="${TEACHER_CONFIG:-configs/distill_teacher.yaml}"
REPORT_DIR="${REPORT_DIR:-reports}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/pretrain_full/best.pt}"

RAW_DIR="data/distill/raw"
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
