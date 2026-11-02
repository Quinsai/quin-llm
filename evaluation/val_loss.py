"""Measure pretraining validation loss for a checkpoint.

Retention is not only about instruction following: a distilled SFT run can
also drift away from the pretraining distribution.  This script evaluates a
checkpoint on in-memory-mapped ``val.bin`` windows and reports cross-entropy
and perplexity.  Run it for the baseline and mixed checkpoints and compare the
two numbers alongside the behavioral evaluation.

Example:
    python -m evaluation.val_loss \
        --ckpt checkpoints/sft_distill_distill_v1/best.pt \
        --data-dir processed_data/pretrain \
        --batches 100 --json-out reports/val_loss_mixed.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pipeline.common import build_model_from_checkpoint, resolve_device
from pipeline.pretrain.dataset import PretrainDataset, load_pretrain_meta
from trainer.tiny_trainer import Trainer


def evaluate_val_loss(
    ckpt: str,
    *,
    data_dir: str = "processed_data/pretrain",
    batch_size: int = 8,
    batches: int | None = 100,
    device: str | None = None,
    amp: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    resolved = resolve_device(device)
    checkpoint = Trainer.load_checkpoint(ckpt, map_location="cpu")
    model = build_model_from_checkpoint(checkpoint, resolved)
    model.eval()

    meta = load_pretrain_meta(data_dir)
    val_path = Path(data_dir) / "val.bin"
    if not val_path.exists():
        raise FileNotFoundError(f"pretraining validation file not found: {val_path}")
    block_size = int(model.config.block_size)
    if int(meta.get("val_tokens", 0)) < block_size + 1:
        raise ValueError(
            f"{val_path} has too few validation tokens for block_size={block_size}"
        )

    dataset = PretrainDataset(
        val_path,
        block_size=block_size,
        dtype=meta["dtype"],
        random_windows=False,
    )
    if batches is not None:
        n_windows = min(len(dataset), max(1, int(batches)) * int(batch_size))
        dataset = torch.utils.data.Subset(dataset, range(n_windows))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    torch.manual_seed(seed)
    total_loss = 0.0
    total_windows = 0
    seen_batches = 0
    use_amp = bool(amp) and resolved.startswith("cuda")
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(resolved)
            y = y.to(resolved)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    _, loss = model(x, y)
            else:
                _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite validation loss at batch {seen_batches}")
            total_loss += float(loss) * x.shape[0]
            total_windows += int(x.shape[0])
            seen_batches += 1
            if batches is not None and seen_batches >= int(batches):
                break

    if not total_windows:
        raise ValueError("no validation windows were evaluated")
    mean_loss = total_loss / total_windows
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "checkpoint_step": checkpoint.get("step"),
        "data_dir": str(data_dir),
        "val_tokens": int(meta.get("val_tokens", 0)),
        "block_size": block_size,
        "batch_size": int(batch_size),
        "batches": seen_batches,
        "windows": total_windows,
        "tokens": total_windows * block_size,
        "val_loss": round(mean_loss, 6),
        "perplexity": round(math.exp(mean_loss), 4),
        "device": resolved,
        "amp": use_amp,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretraining validation loss")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-dir", default="processed_data/pretrain")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--batches",
        type=int,
        default=100,
        help="cap evaluated batches; use -1 for the full validation split",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = evaluate_val_loss(
        args.ckpt,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        batches=None if args.batches < 0 else args.batches,
        device=args.device,
        amp=not args.no_amp,
        seed=args.seed,
    )
    print(
        f"val_loss={result['val_loss']:.6f} "
        f"perplexity={result['perplexity']:.4f} "
        f"({result['windows']:,} windows, step={result['checkpoint_step']})"
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  report: {out}")


if __name__ == "__main__":
    main()
