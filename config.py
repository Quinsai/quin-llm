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
    "TeacherConfig",
    "DistillGenerationConfig",
    "DistillFilterConfig",
    "DatasetBuildConfig",
    "EvalConfig",
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


# ----------------------------------------------------------------------
# teacher distillation
# ----------------------------------------------------------------------
@dataclass
class TeacherConfig(_StageConfig):
    """OpenAI-compatible teacher API settings.

    ``base_url`` and ``model`` may contain ``${ENV_VAR}`` references.  The API
    key is resolved in this order: ``api_key`` (literal in YAML),
    ``api_key_file`` (path to a file containing the key), then the environment
    variable named by ``api_key_env``.  The resolved key is never logged.
    """

    base_url: str = ""
    api_key: str = ""
    api_key_file: str | None = None
    api_key_env: str = "TEACHER_API_KEY"
    model: str = ""
    timeout: float = 60.0
    max_retries: int = 5
    retry_backoff: float = 1.5
    concurrency: int = 8
    requests_per_minute: int | None = 120
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 256
    response_format: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    price_input_per_1k: float = 0.0
    price_output_per_1k: float = 0.0


@dataclass
class DistillGenerationConfig(_StageConfig):
    """Task-driven synthetic data generation settings."""

    run_id: str = "distill_v1"
    teacher_config: str = "configs/distill_teacher.yaml"
    out_dir: str = "data/distill/raw"
    tasks_dir: str = "data/distill/tasks"
    prompt_version: str = "v2"
    seed: int = 42
    max_requests: int | None = None
    max_completion_tokens: int | None = None
    categories: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistillFilterConfig(_StageConfig):
    """Quality filters applied to raw teacher outputs."""

    tokenizer_path: str = "processed_data/tokenizer/tokenizer.pkl"
    block_size: int = 512
    eval_dir: str = "evaluation/datasets"
    eval_entity_seeds: str = "distillation/seeds/eval_entities.jsonl"
    min_instruction_chars: int = 3
    max_instruction_tokens: int = 200
    max_output_tokens: int = 150
    max_output_chars: int = 1200
    min_english_ratio: float = 0.8
    near_dup_threshold: float = 0.85
    max_per_entity: int = 40
    review_fraction: float = 0.05
    min_review_samples: int = 100
    rejected_review_fraction: float = 0.02
    min_rejected_review: int = 20
    seed: int = 42


@dataclass
class DatasetBuildConfig(_StageConfig):
    """Mix teacher data with Alpaca into an SFT parquet."""

    filtered_path: str = ""
    alpaca_paths: str = "data/raw/alpaca/*.parquet"
    tokenizer_path: str = "processed_data/tokenizer/tokenizer.pkl"
    baseline_budget: str = "data/distill/baseline_budget.json"
    out_dir: str = "data/distill/sft/distill_v1"
    teacher_ratio: float = 0.5
    target_samples: int | None = None
    max_steps: int | None = None
    batch_size: int = 32
    seed: int = 42
    write_alpaca_control: bool = False


@dataclass
class EvalConfig(_StageConfig):
    """Checkpoint evaluation settings."""

    ckpt: str = ""
    tokenizer: str | None = None
    device: str | None = None
    eval_dir: str = "evaluation/datasets"
    out: str = "reports/eval.jsonl"
    metrics_out: str | None = None
    max_new_tokens_fact: int = 24
    max_new_tokens_open: int = 200
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0
    limit: int | None = None
    categories: list[str] | None = None
    splits: list[str] | None = None
    seed: int = 42
