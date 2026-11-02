"""CLI for filtering raw teacher responses.

Examples:
    python -m distillation.run_filter \
        --raw data/distill/raw/distill_v1.jsonl \
        --out data/distill/filtered/distill_v1.jsonl
    python -m distillation.run_filter \
        --raw data/distill/raw/smoke.jsonl \
        --out data/distill/filtered/smoke.jsonl --limit 50 --no-contamination
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DistillFilterConfig

from .data_filter import filter_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter teacher-generated SFT data")
    parser.add_argument("--raw", required=True, help="raw JSONL from generation")
    parser.add_argument("--out", required=True, help="filtered JSONL output")
    parser.add_argument("--config", default=None, help="filter YAML")
    parser.add_argument("--rejected-out", default=None)
    parser.add_argument("--stats-out", default=None)
    parser.add_argument("--review-out", default=None)
    parser.add_argument(
        "--no-contamination",
        action="store_true",
        help="skip evaluation-contamination checks (smoke tests only)",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = (
        DistillFilterConfig.from_yaml(args.config)
        if args.config
        else DistillFilterConfig()
    )
    raw = Path(args.raw)
    if not raw.exists():
        raise SystemExit(f"raw file not found: {raw}")
    filter_records(
        raw,
        args.out,
        cfg,
        rejected_path=args.rejected_out,
        stats_path=args.stats_out,
        review_path=args.review_out,
        contamination=not args.no_contamination,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
