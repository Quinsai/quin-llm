#!/usr/bin/env bash
# Download the raw parquet files used by the full configs.
#
# Designed to run ON THE TRAINING SERVER. Defaults to ModelScope, which mirrors
# these datasets file-for-file and is reachable from mainland China without a
# proxy. Set BACKEND=hf to use an HF endpoint instead (e.g. hf-mirror.com).
#
#   ./scripts/download_data.sh                 # ModelScope
#   BACKEND=hf HF_ENDPOINT=https://hf-mirror.com ./scripts/download_data.sh
#   ./scripts/download_data.sh data/raw        # custom destination
#
# The script is resumable: rerun it after an interruption.
set -euo pipefail

BACKEND="${BACKEND:-modelscope}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DEST="${1:-data/raw}"

# Invalidate the completion marker before (re)downloading so an interrupted
# run is never mistaken for a complete one.
rm -f "$DEST/.complete"

# hf_repo|modelscope_repo|remote_path|local_path
FILES=(
  "roneneldan/TinyStories|AI-ModelScope/TinyStories|data/train-00000-of-00004-2d5a1467fff1081b.parquet|tinystories/train-00000-of-00004.parquet"
  "roneneldan/TinyStories|AI-ModelScope/TinyStories|data/train-00001-of-00004-5852b56a2bd28fd9.parquet|tinystories/train-00001-of-00004.parquet"
  "roneneldan/TinyStories|AI-ModelScope/TinyStories|data/train-00002-of-00004-a26307300439e943.parquet|tinystories/train-00002-of-00004.parquet"
  "roneneldan/TinyStories|AI-ModelScope/TinyStories|data/train-00003-of-00004-d243063613e5a057.parquet|tinystories/train-00003-of-00004.parquet"
  "Salesforce/wikitext|Salesforce/wikitext|wikitext-103-raw-v1/train-00000-of-00002.parquet|wikitext103/train-00000-of-00002.parquet"
  "Salesforce/wikitext|Salesforce/wikitext|wikitext-103-raw-v1/train-00001-of-00002.parquet|wikitext103/train-00001-of-00002.parquet"
  "tatsu-lab/alpaca|OmniData/alpaca|data/train-00000-of-00001-a09b74b3ef9c3b56.parquet|alpaca/train.parquet"
)

for item in "${FILES[@]}"; do
  IFS='|' read -r hf_repo ms_repo remote local <<<"$item"
  out="$DEST/$local"
  mkdir -p "$(dirname "$out")"

  case "$BACKEND" in
    hf) url="$HF_ENDPOINT/datasets/$hf_repo/resolve/main/$remote" ;;
    modelscope) url="https://www.modelscope.cn/api/v1/datasets/$ms_repo/repo?Revision=master&FilePath=$remote" ;;
    *) echo "unknown BACKEND=$BACKEND (use modelscope or hf)" >&2; exit 1 ;;
  esac

  echo "==> [$BACKEND] $remote -> $out"
  curl -fL --retry 8 --retry-delay 3 --retry-all-errors -C - -o "$out" "$url"
done

touch "$DEST/.complete"
echo "done: $(du -sh "$DEST" | cut -f1) in $DEST"
