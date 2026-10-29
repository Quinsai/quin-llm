"""Tokenize pretraining corpora into flat uint16/uint32 token files."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from data_pipeline.sources import iter_source_texts


def prepare_pretrain(cfg: dict, tokenizer) -> dict:
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = cfg.get("sources", [])
    if not sources:
        raise ValueError("pretrain config has no sources")

    total_budget = int(cfg["max_tokens"])
    val_fraction = float(cfg.get("val_fraction", 0.01))
    seed = int(cfg.get("seed", 42))
    weights = [float(s.get("weight", 1.0)) for s in sources]
    weight_sum = sum(weights)

    dtype = np.uint16 if tokenizer.vocab_size <= 65_535 else np.uint32
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    counts = {"train": 0, "val": 0}
    buffers = {"train": [], "val": []}
    train_f = open(train_path, "wb")
    val_f = open(val_path, "wb")

    def flush(split: str) -> None:
        if not buffers[split]:
            return
        arr = np.asarray(buffers[split], dtype=dtype)
        (train_f if split == "train" else val_f).write(arr.tobytes())
        buffers[split].clear()

    rng = random.Random(seed)
    written = 0
    try:
        for source, weight in zip(sources, weights):
            budget = int(total_budget * weight / weight_sum)
            used = 0
            print(f"pretrain data: source={source.get('dataset', source.get('path'))} budget={budget:,}")
            for text in iter_source_texts(source):
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if not ids:
                    continue
                split = "val" if (val_fraction > 0 and rng.random() < val_fraction) else "train"
                buffers[split].extend(ids)
                counts[split] += len(ids)
                used += len(ids)
                written += len(ids)
                if len(buffers[split]) >= 2_000_000:
                    flush(split)
                if used >= budget or written >= total_budget:
                    break
            flush("train")
            flush("val")
            print(f"  source done: train={counts['train']:,} val={counts['val']:,}")
            if written >= total_budget:
                break
    finally:
        flush("train")
        flush("val")
        train_f.close()
        val_f.close()

    meta = {
        "tokenizer_path": str(cfg.get("tokenizer_path", "")),
        "vocab_size": tokenizer.vocab_size,
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "max_tokens": total_budget,
        "val_fraction": val_fraction,
        "sources": sources,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(
        f"pretrain data ready: {train_path} ({counts['train']:,} tokens), "
        f"{val_path} ({counts['val']:,} tokens)"
    )
    return meta
