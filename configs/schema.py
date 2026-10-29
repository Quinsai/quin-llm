"""Low-level configuration schemas and YAML helpers.

Stage-specific configuration objects live in the project-level ``config.py``.
This module only contains the shared pieces that are useful everywhere:
``ModelConfig``, ``TrainConfig``, and small YAML/path helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Everything needed to rebuild a TinyModel."""

    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    dropout: float = 0.1
    tie_weights: bool = True
    use_sdpa: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TrainConfig:
    """Shared optimizer / loop settings for pretraining and SFT."""

    out_dir: str = "checkpoints/run"
    batch_size: int = 64
    grad_accum_steps: int = 1
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 500
    max_steps: int | None = 20_000
    epochs: int | None = None
    eval_interval: int = 250
    eval_batches: int = 50
    save_interval: int = 1_000
    log_interval: int = 20
    amp: bool = True
    num_workers: int = 4
    seed: int = 42
    draw: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
