"""Orchestration for SFT data preparation and fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

from config import ModelConfig, PrepareSFTConfig, SFTConfig, SFTDataConfig
from model.model import TinyModel
from pipeline.common import load_tokenizer
from pipeline.pretrain.dataset import PretrainDataset, load_pretrain_meta
from pipeline.sft.dataset import ReplayDataset, SFTDataset
from pipeline.sft.prepare import prepare_sft
from trainer.tiny_trainer import Trainer
from utils.seed import set_seed


def _sft_block_size(data_cfg: SFTDataConfig, ckpt: str | None) -> int:
    if ckpt:
        model_config = Trainer.load_model_config(ckpt)
        return int(model_config["block_size"])
    return int(data_cfg.block_size)


def run_prepare_sft(cfg: PrepareSFTConfig, ckpt: str | None = None) -> None:
    data_cfg = cfg.data
    tokenizer = load_tokenizer(data_cfg.tokenizer_path)
    block_size = _sft_block_size(data_cfg, ckpt)
    prepare_sft(data_cfg.to_dict(), tokenizer, block_size)


def run_sft(
    cfg: SFTConfig,
    ckpt: str,
    epochs: int | None = None,
    resume_from: str | None = None,
) -> float:
    data_cfg = cfg.data
    train_cfg = cfg.train
    set_seed(train_cfg.seed)

    checkpoint = Trainer.load_checkpoint(ckpt)
    model_cfg = ModelConfig.from_dict(checkpoint["model_config"])
    tokenizer_path = data_cfg.tokenizer_path or checkpoint.get("tokenizer_path")
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} does not match "
            f"checkpoint vocab_size={model_cfg.vocab_size}"
        )

    out_dir = Path(data_cfg.out_dir)
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            sft_meta = json.load(f)
        if "total_tokens" in sft_meta:
            print(
                f"SFT tokens: train={int(sft_meta['train_total_tokens']):,}, "
                f"val={int(sft_meta['val_total_tokens']):,}, "
                f"answer={int(sft_meta['answer_tokens']):,} "
                f"(avg total={float(sft_meta['avg_total_tokens']):.1f})"
            )
    train_ds = SFTDataset(out_dir / "train.pt")
    val_ds = SFTDataset(out_dir / "val.pt") if (out_dir / "val.pt").exists() else None
    if val_ds is not None and len(val_ds) == 0:
        val_ds = None

    replay_prob = float(data_cfg.replay_prob)
    if replay_prob > 0:
        pretrain_dir = Path(data_cfg.pretrain_dir)
        replay_meta = load_pretrain_meta(pretrain_dir)
        replay_ds = PretrainDataset(
            pretrain_dir / "train.bin",
            block_size=model_cfg.block_size,
            dtype=replay_meta["dtype"],
            random_windows=True,
            seed=train_cfg.seed,
        )
        train_ds = ReplayDataset(train_ds, replay_ds, replay_prob)
        print(f"SFT replay enabled: {replay_prob:.0%} pretrain windows")
    print(f"SFT samples: train={len(train_ds):,}, val={len(val_ds) if val_ds else 0:,}")

    model = TinyModel(model_cfg)
    trainer = Trainer(model, tokenizer, train_cfg, tokenizer_path=tokenizer_path)
    return trainer.train_sft(
        train_ds,
        val_ds,
        epochs=epochs,
        pretrain_checkpoint=ckpt,
        resume_from=resume_from,
    )
