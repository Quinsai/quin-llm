"""CLI for mixing filtered teacher data with Alpaca.

Examples:
    python -m distillation.run_build_dataset \
        --config configs/distill_dataset.yaml
    python -m distillation.run_build_dataset \
        --filtered data/distill/filtered/distill_v1.jsonl \
        --out data/distill/sft/distill_v1 --teacher-ratio 0.5
"""

from __future__ import annotations

import argparse

from config import DatasetBuildConfig

from .dataset_builder import run_build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed SFT parquet")
    parser.add_argument("--config", default=None, help="dataset build YAML")
    parser.add_argument("--filtered", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--alpaca-paths", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--baseline-budget", default=None)
    parser.add_argument("--teacher-ratio", type=float, default=None)
    parser.add_argument("--target-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--write-alpaca-control",
        action="store_true",
        help="also write an Alpaca-only dataset with the same row count",
    )
    args = parser.parse_args()

    cfg = (
        DatasetBuildConfig.from_yaml(args.config)
        if args.config
        else DatasetBuildConfig()
    )
    if args.filtered:
        cfg.filtered_path = args.filtered
    if args.out:
        cfg.out_dir = args.out
    if args.alpaca_paths:
        cfg.alpaca_paths = args.alpaca_paths
    if args.tokenizer:
        cfg.tokenizer_path = args.tokenizer
    if args.baseline_budget:
        cfg.baseline_budget = args.baseline_budget
    if args.teacher_ratio is not None:
        cfg.teacher_ratio = args.teacher_ratio
    if args.target_samples is not None:
        cfg.target_samples = args.target_samples
    if args.seed is not None:
        cfg.seed = args.seed
    if args.write_alpaca_control:
        cfg.write_alpaca_control = True

    run_build_dataset(cfg)


if __name__ == "__main__":
    main()
