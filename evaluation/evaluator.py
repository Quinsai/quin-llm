"""Load a checkpoint, run frozen evaluation prompts, and score the outputs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from config import EvalConfig
from pipeline.common import build_model_from_checkpoint, load_tokenizer, resolve_device
from pipeline.sft.prepare import PROMPT_TEMPLATE
from trainer.tiny_trainer import Trainer
from utils.tqdm_utils import PROGRESS_BAR_FORMAT, progress_kwargs

from .metrics import repetition_rate, score_prediction, summarize_records


def load_eval_examples(
    eval_dir: str | Path,
    *,
    categories: list[str] | None = None,
    splits: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    root = Path(eval_dir)
    if not root.exists():
        raise FileNotFoundError(f"evaluation directory not found: {root}")
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise ValueError(f"no evaluation JSONL files under {root}")

    examples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                sample_id = str(record.get("id", "")).strip()
                if not sample_id:
                    raise ValueError(f"{path}:{line_no}: missing id")
                if sample_id in seen:
                    raise ValueError(f"duplicate evaluation id: {sample_id!r}")
                seen.add(sample_id)
                record = dict(record)
                record["category"] = str(record.get("category", ""))
                record["split"] = str(record.get("split", ""))
                record["check"] = str(record.get("check", "alias_match"))
                record["_file"] = str(path)
                examples.append(record)

    if categories:
        allowed = set(categories)
        examples = [e for e in examples if e["category"] in allowed]
    if splits:
        allowed = set(splits)
        examples = [e for e in examples if e["split"] in allowed]
    examples.sort(key=lambda e: (e["category"], e["split"], str(e["id"])))
    if limit is not None:
        examples = examples[: max(0, int(limit))]
    if not examples:
        raise ValueError("no evaluation examples selected")
    return examples


def _prompt_for(example: dict[str, Any]) -> str:
    user = str(example.get("instruction", "")).strip()
    extra = str(example.get("input", "")).strip()
    if extra:
        user = user + "\n" + extra
    return PROMPT_TEMPLATE.format(user=user)


def _generate(
    model,
    tokenizer,
    device: str,
    prompt: str,
    *,
    max_new_tokens: int,
    cfg: EvalConfig,
) -> tuple[str, int, bool, int]:
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    block_size = int(model.config.block_size)
    truncated = False
    if len(ids) > block_size - 1:
        ids = ids[-(block_size - 1) :]
        truncated = True
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.generate(
            idx,
            max_new_tokens=int(max_new_tokens),
            temperature=float(cfg.temperature),
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            repetition_penalty=float(cfg.repetition_penalty),
            eos_token_id=tokenizer.eos_id,
        )
    new_ids = out[0, len(ids) :].tolist()
    return tokenizer.decode(new_ids).strip(), len(new_ids), truncated, len(ids)


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    if not cfg.ckpt:
        raise ValueError("ckpt is required")
    device = resolve_device(cfg.device)
    checkpoint = Trainer.load_checkpoint(cfg.ckpt, map_location="cpu")
    tokenizer_path = cfg.tokenizer or checkpoint.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("checkpoint has no tokenizer_path; pass --tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model_from_checkpoint(checkpoint, device)
    print(f"model: {model.num_parameters():,} params on {device}")

    examples = load_eval_examples(
        cfg.eval_dir,
        categories=cfg.categories,
        splits=cfg.splits,
        limit=cfg.limit,
    )
    print(f"evaluation: {len(examples):,} example(s) from {cfg.eval_dir}")

    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(cfg.metrics_out)
        if cfg.metrics_out
        else out_path.with_suffix(".metrics.json")
    )

    records: list[dict[str, Any]] = []
    with open(out_path, "w", encoding="utf-8") as f, tqdm(
        total=len(examples),
        desc="eval",
        unit="sample",
        dynamic_ncols=True,
        bar_format=PROGRESS_BAR_FORMAT,
        **progress_kwargs(),
    ) as pbar:
        for example in examples:
            category = example["category"]
            default_new = (
                cfg.max_new_tokens_fact
                if category == "fact_qa"
                else cfg.max_new_tokens_open
            )
            max_new = int(example.get("max_new_tokens") or default_new)
            prompt = _prompt_for(example)
            started = time.time()
            prediction, generated, truncated, prompt_tokens = _generate(
                model,
                tokenizer,
                device,
                prompt,
                max_new_tokens=max_new,
                cfg=cfg,
            )
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.time() - started
            score, correct, detail = score_prediction(example, prediction)
            record = {
                "id": example["id"],
                "category": category,
                "split": example["split"],
                "instruction": example.get("instruction", ""),
                "reference": example.get("reference"),
                "aliases": example.get("aliases"),
                "check": example.get("check", ""),
                "prediction": prediction,
                "score": score,
                "correct": correct,
                "detail": detail,
                "repetition_rate": round(repetition_rate(prediction), 4),
                "generated_tokens": generated,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new,
                "prompt_truncated": truncated,
                "elapsed_sec": round(elapsed, 3),
                "checkpoint": str(cfg.ckpt),
                "eval_file": example.get("_file", ""),
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            pbar.update(1)

    metrics = summarize_records(records, seed=cfg.seed)
    metrics["metadata"] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(cfg.ckpt),
        "checkpoint_step": checkpoint.get("step"),
        "tokenizer": str(tokenizer_path),
        "device": device,
        "eval_dir": str(cfg.eval_dir),
        "out": str(out_path),
        "examples": len(records),
        "config": cfg.to_dict(),
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    overall = metrics["overall"]
    print(
        f"eval: accuracy={overall['accuracy']} "
        f"score={overall['mean_score']} "
        f"scored={overall['scored']}/{overall['n']}"
    )
    for category, summary in metrics["by_category"].items():
        print(
            f"  {category:16s} n={summary['n']:3d} "
            f"acc={summary['accuracy']} score={summary['mean_score']}"
        )
    print(f"  raw outputs: {out_path}")
    print(f"  metrics: {metrics_path}")
    return metrics
