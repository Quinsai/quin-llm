#!/usr/bin/env bash
# Local teacher-API smoke test. No GPU, checkpoint, or training data needed.
#
# 1. resolve credentials from TEACHER_CONFIG
# 2. send one tiny ping
# 3. send one real generation request and write the raw response
#
# Usage:
#   ./scripts/distill/99_api_smoke.sh
#   ./scripts/distill/99_api_smoke.sh --limit 5
#   ./scripts/distill/99_api_smoke.sh --categories fact_qa,format_json --limit 10
#   ./scripts/distill/99_api_smoke.sh --dry-run --limit 5
set -euo pipefail
source "$(dirname "$0")/_env.sh"

export TEACHER_CONFIG
SMOKE_RUN_ID="${SMOKE_RUN_ID:-api_smoke}"
mkdir -p "$RAW_DIR"

echo "==> teacher ping"
"$PYTHON" - <<'PY'
import os

from config import TeacherConfig
from distillation.teacher_client import TeacherClient, api_key_source

cfg = TeacherConfig.from_yaml(os.environ["TEACHER_CONFIG"])
client = TeacherClient(cfg)
print(f"endpoint: {client.base_url}")
print(f"model:    {client.model}")
print(f"key:      {api_key_source(cfg)}")
print(f"ping:     {client.ping()!r}")
PY

echo
echo "==> one generation request"
"$PYTHON" -m distillation.run_generation \
    --config configs/distill_generation.yaml \
    --teacher-config "$TEACHER_CONFIG" \
    --run-id "$SMOKE_RUN_ID" \
    --limit 1 \
    "$@"

echo
echo "raw response: ${RAW_DIR}/${SMOKE_RUN_ID}.jsonl"
