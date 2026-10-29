"""Stage-specific configuration objects for quin-llm.

Each pipeline stage reads the section of the YAML it needs through a typed
dataclass.  The low-level model/training schemas and YAML helpers live in
``configs/schema.py``; they are re-exported here so existing imports keep
working.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from configs.schema import ModelConfig, TrainConfig, ensure_dir, load_yaml

__all__ = [
    "ModelConfig",
    "TrainConfig",
    "load_yaml",
    "ensure_dir",
    "TokenizerConfig",
    "PretrainDataConfig",
    "PreparePretrainConfig",
    "PretrainConfig",
    "SFTDataConfig",
    "PrepareSFTConfig",
    "SFTConfig",
    "InferenceConfig",
]


class _StageConfig:
    """Small dataclass mixin for YAML/dict conversion."""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        data = data or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path):
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TokenizerConfig(_StageConfig):
    """Configuration for the byte-level BPE tokenizer."""

    out_path: str = "processed_data/tokenizer/tokenizer.pkl"
    vocab_size: int = 8192
    min_freq: int = 2
    max_words: int = 200_000
    max_chars: int | None = 100_000_000
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PretrainDataConfig(_StageConfig):
    """Data preparation settings shared by prepare-pretrain and pretrain."""

    tokenizer_path: str = ""
    out_dir: str = "processed_data/pretrain"
    max_tokens: int = 300_000_000
    val_fraction: float = 0.01
    seed: int = 42
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparePretrainConfig(_StageConfig):
    """Configuration for ``main.py prepare-pretrain``."""

    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        raw = data or {}
        return cls(data=PretrainDataConfig.from_dict(raw.get("data", raw)))


@dataclass
class PretrainConfig(_StageConfig):
    """Configuration for ``main.py pretrain``."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        raw = data or {}
        return cls(
            model=ModelConfig.from_dict(raw.get("model", {})),
            data=PretrainDataConfig.from_dict(raw.get("data", {})),
            train=TrainConfig.from_dict(raw.get("train", {})),
        )


@dataclass
class SFTDataConfig(_StageConfig):
    """Data preparation settings shared by prepare-sft and sft."""

    tokenizer_path: str = ""
    out_dir: str = "processed_data/sft"
    val_fraction: float = 0.02
    seed: int = 42
    replay_prob: float = 0.0
    pretrain_dir: str = "processed_data/pretrain"
    block_size: int = 512
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrepareSFTConfig(_StageConfig):
    """Configuration for ``main.py prepare-sft``."""

    data: SFTDataConfig = field(default_factory=SFTDataConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        raw = data or {}
        return cls(data=SFTDataConfig.from_dict(raw.get("data", raw)))


@dataclass
class SFTConfig(_StageConfig):
    """Configuration for ``main.py sft``."""

    data: SFTDataConfig = field(default_factory=SFTDataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        raw = data or {}
        return cls(
            data=SFTDataConfig.from_dict(raw.get("data", {})),
            train=TrainConfig.from_dict(raw.get("train", {})),
        )


@dataclass
class InferenceConfig(_StageConfig):
    """Configuration for generate/chat inference."""

    ckpt: str = ""
    tokenizer: str | None = None
    device: str | None = None
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float | None = 0.95
    repetition_penalty: float = 1.1

    @classmethod
    def from_args(cls, args) -> "InferenceConfig":
        return cls(
            ckpt=args.ckpt,
            tokenizer=args.tokenizer,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
