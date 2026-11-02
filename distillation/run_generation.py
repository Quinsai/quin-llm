"""CLI for teacher data generation.

Examples:
    python -m distillation.run_generation \
        --config configs/distill_generation.yaml --dry-run --limit 5
    python -m distillation.run_generation \
        --config configs/distill_generation.yaml --limit 5 --run-id smoke
    python -m distillation.run_generation \
        --config configs/distill_generation.yaml
"""

from __future__ import annotations

import argparse

from config import DistillGenerationConfig

from .data_generator import run_generation


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SFT data with a teacher API")
    parser.add_argument("--config", required=True, help="distillation generation YAML")
    parser.add_argument("--run-id", default=None, help="override config run_id")
    parser.add_argument(
        "--teacher-config",
        default=None,
        help="override the teacher YAML (e.g. a local file containing api_key)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print prompts only")
    parser.add_argument("--limit", type=int, default=None, help="max new requests")
    parser.add_argument(
        "--reuse-tasks",
        action="store_true",
        help="reuse tasks/{run_id}.jsonl instead of rebuilding it",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="comma-separated category filter, e.g. fact_qa,format_json",
    )
    args = parser.parse_args()

    cfg = DistillGenerationConfig.from_yaml(args.config)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.teacher_config:
        cfg.teacher_config = args.teacher_config
    categories = None
    if args.categories:
        categories = [x.strip() for x in args.categories.split(",") if x.strip()]

    run_generation(
        cfg,
        dry_run=args.dry_run,
        limit=args.limit,
        reuse_tasks=args.reuse_tasks,
        categories=categories,
    )


if __name__ == "__main__":
    main()
