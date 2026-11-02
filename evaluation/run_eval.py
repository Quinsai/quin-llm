"""CLI for checkpoint evaluation.

Example:
    python -m evaluation.run_eval \
        --ckpt checkpoints/sft_distill/best.pt \
        --eval-dir evaluation/datasets \
        --out reports/eval_mixed.jsonl \
        --metrics-out reports/eval_mixed.metrics.json
"""

from __future__ import annotations

import argparse

from config import EvalConfig

from .evaluator import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a quin-llm checkpoint")
    parser.add_argument("--config", default=None, help="evaluation YAML")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--categories", default=None, help="comma-separated")
    parser.add_argument("--splits", default=None, help="comma-separated")
    parser.add_argument("--max-new-tokens-fact", type=int, default=None)
    parser.add_argument("--max-new-tokens-open", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = EvalConfig.from_yaml(args.config) if args.config else EvalConfig()
    if args.ckpt:
        cfg.ckpt = args.ckpt
    if args.tokenizer:
        cfg.tokenizer = args.tokenizer
    if args.device:
        cfg.device = args.device
    if args.eval_dir:
        cfg.eval_dir = args.eval_dir
    if args.out:
        cfg.out = args.out
    if args.metrics_out:
        cfg.metrics_out = args.metrics_out
    if args.limit is not None:
        cfg.limit = args.limit
    if args.categories:
        cfg.categories = [x.strip() for x in args.categories.split(",") if x.strip()]
    if args.splits:
        cfg.splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if args.max_new_tokens_fact is not None:
        cfg.max_new_tokens_fact = args.max_new_tokens_fact
    if args.max_new_tokens_open is not None:
        cfg.max_new_tokens_open = args.max_new_tokens_open
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.repetition_penalty is not None:
        cfg.repetition_penalty = args.repetition_penalty
    if args.seed is not None:
        cfg.seed = args.seed

    if not cfg.ckpt:
        raise SystemExit("--ckpt or ckpt in config is required")
    run_eval(cfg)


if __name__ == "__main__":
    main()
