"""Shared helpers used by pipeline stages and inference."""

from __future__ import annotations

from pathlib import Path

import torch

from config import ModelConfig
from model.model import TinyModel
from pipeline.tokenizer.bpe_tokenizer import BPETokenizer


def load_tokenizer(path: str | Path) -> BPETokenizer:
    tokenizer = BPETokenizer.load(path)
    print(f"tokenizer: {path} (vocab_size={tokenizer.vocab_size})")
    return tokenizer


def resolve_device(name: str | None) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model_from_checkpoint(checkpoint: dict, device: str) -> TinyModel:
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = TinyModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model
