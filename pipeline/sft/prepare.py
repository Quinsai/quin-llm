"""Tokenize instruction data with token-level answer masking."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pipeline.sources import iter_sft_examples, parse_sft_file

PROMPT_TEMPLATE = "###User:\n{user}\n###Agent:\n"


def prepare_sft(cfg: dict, tokenizer, block_size: int) -> dict:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    source = cfg.get("source")
    if not source:
        raise ValueError("SFT config has no source")

    val_fraction = float(cfg.get("val_fraction", 0.02))
    seed = int(cfg.get("seed", 42))
    pad_id = tokenizer.pad_id

    # Store one extra token per sample.  The training model consumes
    # ``block_size`` inputs and predicts the next token, so ``SFTDataset``
    # returns shifted inputs/targets.  Files keep a fixed width for simple
    # storage; ``SFTDataset`` trims the padding and the loader pads each
    # batch to its own longest sample.
    stored_len = int(block_size) + 1

    samples: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0
    overlong = 0

    total_examples = source.get("num_examples") or source.get("total")
    if total_examples is None and source.get("type", "hf") == "file":
        # Local files are cheap to scan once, so show a real percentage.
        total_examples = sum(1 for _ in parse_sft_file(source["path"]))
    with tqdm(
        total=total_examples,
        desc="prepare-sft",
        unit="sample",
        dynamic_ncols=True,
    ) as pbar:
        for user, answer in iter_sft_examples(source):
            pbar.update(1)
            prompt = PROMPT_TEMPLATE.format(user=user)
            prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
            answer_ids = tokenizer.encode(answer, add_bos=False, add_eos=True)
            if len(answer_ids) <= 1:
                # Only EOS: the source produced an empty answer.
                skipped += 1
                pbar.set_postfix(kept=len(samples), skipped=skipped, overlong=overlong)
                continue
            if len(prompt_ids) + len(answer_ids) > stored_len:
                # Truncation would either drop the instruction or cut the
                # answer mid-sentence and force a premature EOS.  Skip instead.
                skipped += 1
                overlong += 1
                pbar.set_postfix(kept=len(samples), skipped=skipped, overlong=overlong)
                continue
            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids
            pad = stored_len - len(ids)
            ids = ids + [pad_id] * pad
            labels = labels + [-100] * pad
            samples.append(
                (
                    np.asarray(ids, dtype=np.int32),
                    np.asarray(labels, dtype=np.int32),
                )
            )
            pbar.set_postfix(kept=len(samples), skipped=skipped, overlong=overlong)

    if not samples:
        raise ValueError("no SFT samples were produced")

    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_fraction)) if val_fraction > 0 else 0
    n_val = min(n_val, max(0, len(samples) - 1))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    if not train_samples:
        raise ValueError("no training samples left after the validation split")

    def pack(part: list[tuple[np.ndarray, np.ndarray]]):
        input_ids = torch.from_numpy(np.stack([x for x, _ in part]))
        labels = torch.from_numpy(np.stack([y for _, y in part]))
        return {"input_ids": input_ids, "labels": labels}

    train = pack(train_samples)
    val = pack(val_samples) if val_samples else {
        "input_ids": torch.empty(0, stored_len, dtype=torch.int32),
        "labels": torch.empty(0, stored_len, dtype=torch.int32),
    }
    torch.save(train, out_dir / "train.pt")
    torch.save(val, out_dir / "val.pt")

    def token_stats(part: dict) -> dict[str, int]:
        labels = part["labels"]
        if labels.size(0) == 0:
            return {
                "samples": 0,
                "total_tokens": 0,
                "answer_tokens": 0,
                "max_total_tokens": 0,
                "max_answer_tokens": 0,
            }
        # Prompt and padding labels are -100; the last answer token is the
        # final non-ignored position, so it also marks the real sequence end.
        answer_mask = labels != -100
        answer_counts = answer_mask.sum(dim=1)
        positions = torch.arange(labels.size(1), dtype=torch.long).unsqueeze(0)
        total_lengths = (answer_mask * positions).max(dim=1).values + 1
        return {
            "samples": int(labels.size(0)),
            "total_tokens": int(total_lengths.sum().item()),
            "answer_tokens": int(answer_counts.sum().item()),
            "max_total_tokens": int(total_lengths.max().item()),
            "max_answer_tokens": int(answer_counts.max().item()),
        }

    train_stats = token_stats(train)
    val_stats = token_stats(val)
    total_tokens = train_stats["total_tokens"] + val_stats["total_tokens"]
    answer_tokens = train_stats["answer_tokens"] + val_stats["answer_tokens"]
    sample_count = train_stats["samples"] + val_stats["samples"]

    meta = {
        "tokenizer_path": str(cfg.get("tokenizer_path", "")),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "skipped": skipped,
        "overlong_samples": overlong,
        "train_total_tokens": train_stats["total_tokens"],
        "val_total_tokens": val_stats["total_tokens"],
        "train_answer_tokens": train_stats["answer_tokens"],
        "val_answer_tokens": val_stats["answer_tokens"],
        "total_tokens": total_tokens,
        "answer_tokens": answer_tokens,
        "avg_total_tokens": total_tokens / sample_count,
        "max_total_tokens": max(
            train_stats["max_total_tokens"], val_stats["max_total_tokens"]
        ),
        "avg_answer_tokens": answer_tokens / sample_count,
        "max_answer_tokens": max(
            train_stats["max_answer_tokens"], val_stats["max_answer_tokens"]
        ),
        "source": source,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(
        f"SFT data ready: train={len(train_samples):,}, val={len(val_samples):,}, "
        f"skipped={skipped:,} (overlong={overlong:,})"
    )
    print(
        f"  tokens: train={train_stats['total_tokens']:,}, "
        f"val={val_stats['total_tokens']:,}, answer={answer_tokens:,} "
        f"(avg total={meta['avg_total_tokens']:.1f}, "
        f"avg answer={meta['avg_answer_tokens']:.1f})"
    )
    return meta
