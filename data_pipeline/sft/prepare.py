"""Tokenize instruction data with token-level answer masking."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_pipeline.sources import iter_sft_examples, parse_sft_file

PROMPT_TEMPLATE = "###User:\n{user}\n###Agent:\n"


def _fit_to_block_size(
    prompt_ids: list[int],
    answer_ids: list[int],
    block_size: int,
    eos_id: int,
    min_prompt_tokens: int = 16,
) -> tuple[list[int], list[int]]:
    """Truncate a sample while keeping the answer (and at least part of prompt).

    The returned arrays may contain ``block_size`` tokens.  The caller stores
    one extra token so the dataset can produce shifted next-token targets.
    """
    if len(prompt_ids) + len(answer_ids) <= block_size:
        return prompt_ids + answer_ids, [-100] * len(prompt_ids) + list(answer_ids)

    min_prompt = min(len(prompt_ids), min_prompt_tokens)
    max_answer = max(1, block_size - min_prompt)
    if len(answer_ids) > max_answer:
        answer_ids = list(answer_ids[:max_answer])
        answer_ids[-1] = eos_id

    keep_prompt = block_size - len(answer_ids)
    if keep_prompt <= 0:
        prompt_ids = []
    elif len(prompt_ids) > keep_prompt:
        if keep_prompt == 1:
            prompt_ids = prompt_ids[:1]
        else:
            # Preserve BOS / the first prompt token, drop from the left.
            prompt_ids = prompt_ids[:1] + prompt_ids[-(keep_prompt - 1) :]

    ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return ids, labels


def prepare_sft(cfg: dict, tokenizer, block_size: int) -> dict:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    source = cfg.get("source")
    if not source:
        raise ValueError("SFT config has no source")

    val_fraction = float(cfg.get("val_fraction", 0.02))
    seed = int(cfg.get("seed", 42))
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id

    # Store one extra token per sample.  The training model consumes
    # ``block_size`` inputs and predicts the next token, so ``SFTDataset``
    # returns ``input_ids[:-1]`` and ``labels[1:]``.
    stored_len = int(block_size) + 1

    samples: list[tuple[np.ndarray, np.ndarray]] = []
    skipped = 0
    answer_lengths: list[int] = []

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
            if not answer_ids:
                skipped += 1
                pbar.set_postfix(kept=len(samples), skipped=skipped)
                continue
            ids, labels = _fit_to_block_size(prompt_ids, answer_ids, stored_len, eos_id)
            if len(ids) > stored_len:
                # Should not happen, but never write malformed samples.
                skipped += 1
                pbar.set_postfix(kept=len(samples), skipped=skipped)
                continue
            pad = stored_len - len(ids)
            ids = ids + [pad_id] * pad
            labels = labels + [-100] * pad
            samples.append(
                (
                    np.asarray(ids, dtype=np.int32),
                    np.asarray(labels, dtype=np.int32),
                )
            )
            answer_lengths.append(sum(1 for x in labels if x != -100))
            pbar.set_postfix(kept=len(samples), skipped=skipped)

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

    meta = {
        "tokenizer_path": str(cfg.get("tokenizer_path", "")),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "skipped": skipped,
        "avg_answer_tokens": float(np.mean(answer_lengths)),
        "max_answer_tokens": int(np.max(answer_lengths)),
        "source": source,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(
        f"SFT data ready: train={len(train_samples):,}, val={len(val_samples):,}, "
        f"skipped={skipped:,}, avg_answer_tokens={meta['avg_answer_tokens']:.1f}"
    )
    return meta
