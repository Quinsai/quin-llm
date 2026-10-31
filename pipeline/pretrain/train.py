"""Orchestration for pretraining data preparation and training."""

from __future__ import annotations

from pathlib import Path

from config import PreparePretrainConfig, PretrainConfig
from model.model import TinyModel
from pipeline.common import load_tokenizer
from pipeline.pretrain.dataset import PretrainDataset, load_pretrain_meta
from pipeline.pretrain.prepare import prepare_pretrain
from trainer.tiny_trainer import Trainer
from utils.seed import set_seed


def run_prepare_pretrain(cfg: PreparePretrainConfig) -> None:
    data_cfg = cfg.data
    tokenizer = load_tokenizer(data_cfg.tokenizer_path)
    prepare_pretrain(data_cfg.to_dict(), tokenizer)


def run_pretrain(cfg: PretrainConfig, resume_from: str | None = None) -> float:
    data_cfg = cfg.data
    train_cfg = cfg.train
    set_seed(train_cfg.seed)

    tokenizer = load_tokenizer(data_cfg.tokenizer_path)
    model_cfg = cfg.model
    if model_cfg.vocab_size != tokenizer.vocab_size:
        print(
            f"warning: model vocab_size={model_cfg.vocab_size} but tokenizer has "
            f"{tokenizer.vocab_size}; using the tokenizer size"
        )
        model_cfg.vocab_size = tokenizer.vocab_size

    meta = load_pretrain_meta(data_cfg.out_dir)
    train_path = Path(data_cfg.out_dir) / "train.bin"
    val_path = Path(data_cfg.out_dir) / "val.bin"
    train_ds = PretrainDataset(
        train_path,
        block_size=model_cfg.block_size,
        dtype=meta["dtype"],
        random_windows=True,
        seed=train_cfg.seed,
    )
    val_ds = (
        PretrainDataset(
            val_path,
            block_size=model_cfg.block_size,
            dtype=meta["dtype"],
            random_windows=False,
        )
        if meta.get("val_tokens", 0) >= model_cfg.block_size + 1
        else None
    )
    print(
        f"pretrain tokens: train={meta['train_tokens']:,}, "
        f"val={meta['val_tokens']:,}, block_size={model_cfg.block_size}"
    )

    model = TinyModel(model_cfg)
    trainer = Trainer(
        model,
        tokenizer,
        train_cfg,
        tokenizer_path=data_cfg.tokenizer_path,
    )
    return trainer.train_pretrain(train_ds, val_ds, resume_from=resume_from)
