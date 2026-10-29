"""Command line entry point for quin-llm.

Examples:
    python main.py tokenize --config configs/tokenizer.yaml
    python main.py prepare-pretrain --config configs/pretrain_full.yaml
    python main.py pretrain --config configs/pretrain_full.yaml
    python main.py prepare-sft --config configs/sft_full.yaml --ckpt checkpoints/pretrain_full/best.pt
    python main.py sft --config configs/sft_full.yaml --ckpt checkpoints/pretrain_full/best.pt
    python main.py generate --ckpt checkpoints/sft_full/best.pt --prompt "What is a cat?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from config import (
    InferenceConfig,
    ModelConfig,
    PreparePretrainConfig,
    PrepareSFTConfig,
    PretrainConfig,
    SFTConfig,
    SFTDataConfig,
    TokenizerConfig,
)
from data_pipeline.pretrain.dataset import PretrainDataset, load_pretrain_meta
from data_pipeline.pretrain.prepare import prepare_pretrain
from data_pipeline.sft.dataset import ReplayDataset, SFTDataset
from data_pipeline.sft.prepare import PROMPT_TEMPLATE, prepare_sft
from data_pipeline.tokenizer.bpe_tokenizer import BPETokenizer
from data_pipeline.tokenizer.train import train_tokenizer
from model.model import TinyModel
from trainer.tiny_trainer import Trainer
from utils.seed import set_seed


def _load_tokenizer(path: str | Path) -> BPETokenizer:
    tokenizer = BPETokenizer.load(path)
    print(f"tokenizer: {path} (vocab_size={tokenizer.vocab_size})")
    return tokenizer


def _resolve_device(name: str | None) -> str:
    if name:
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model_from_checkpoint(checkpoint: dict, device: str) -> TinyModel:
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = TinyModel(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------
def cmd_tokenize(args) -> None:
    train_tokenizer(TokenizerConfig.from_yaml(args.config))


# ----------------------------------------------------------------------
# pretraining
# ----------------------------------------------------------------------
def cmd_prepare_pretrain(args) -> None:
    cfg = PreparePretrainConfig.from_yaml(args.config)
    data_cfg = cfg.data
    tokenizer = _load_tokenizer(data_cfg.tokenizer_path)
    prepare_pretrain(data_cfg.to_dict(), tokenizer)


def cmd_pretrain(args) -> None:
    cfg = PretrainConfig.from_yaml(args.config)
    data_cfg = cfg.data
    train_cfg = cfg.train
    set_seed(train_cfg.seed)

    tokenizer = _load_tokenizer(data_cfg.tokenizer_path)
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
    trainer.train_pretrain(train_ds, val_ds, resume_from=args.resume)


# ----------------------------------------------------------------------
# supervised fine-tuning
# ----------------------------------------------------------------------
def _sft_block_size(args, data_cfg: SFTDataConfig) -> int:
    if args.ckpt:
        checkpoint = Trainer.load_checkpoint(args.ckpt)
        return int(checkpoint["model_config"]["block_size"])
    return int(data_cfg.block_size)


def cmd_prepare_sft(args) -> None:
    cfg = PrepareSFTConfig.from_yaml(args.config)
    data_cfg = cfg.data
    tokenizer = _load_tokenizer(data_cfg.tokenizer_path)
    block_size = _sft_block_size(args, data_cfg)
    prepare_sft(data_cfg.to_dict(), tokenizer, block_size)


def cmd_sft(args) -> None:
    cfg = SFTConfig.from_yaml(args.config)
    data_cfg = cfg.data
    train_cfg = cfg.train
    set_seed(train_cfg.seed)

    checkpoint = Trainer.load_checkpoint(args.ckpt)
    model_cfg = ModelConfig.from_dict(checkpoint["model_config"])
    tokenizer_path = data_cfg.tokenizer_path or checkpoint.get("tokenizer_path")
    tokenizer = _load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size != model_cfg.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} does not match "
            f"checkpoint vocab_size={model_cfg.vocab_size}"
        )

    out_dir = Path(data_cfg.out_dir)
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
    trainer.train_sft(
        train_ds,
        val_ds,
        epochs=args.epochs,
        pretrain_checkpoint=args.ckpt,
        resume_from=args.resume,
    )


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
def _load_for_inference(cfg: InferenceConfig):
    device = _resolve_device(cfg.device)
    checkpoint = Trainer.load_checkpoint(cfg.ckpt, map_location="cpu")
    tokenizer_path = cfg.tokenizer or checkpoint.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("no tokenizer path in checkpoint; pass --tokenizer")
    tokenizer = _load_tokenizer(tokenizer_path)
    model = _build_model_from_checkpoint(checkpoint, device)
    print(f"model: {model.num_parameters():,} params on {device}")
    return model, tokenizer, device


def _generate(model, tokenizer, device, prompt: str, cfg: InferenceConfig) -> str:
    text = PROMPT_TEMPLATE.format(user=prompt)
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        eos_token_id=tokenizer.eos_id,
    )
    new_ids = out[0, len(ids) :].tolist()
    return tokenizer.decode(new_ids).strip()


def cmd_generate(args) -> None:
    cfg = InferenceConfig.from_args(args)
    model, tokenizer, device = _load_for_inference(cfg)
    answer = _generate(model, tokenizer, device, args.prompt, cfg)
    print(answer)


def cmd_chat(args) -> None:
    cfg = InferenceConfig.from_args(args)
    model, tokenizer, device = _load_for_inference(cfg)
    print("Enter a prompt, or 'quit' to exit.")
    while True:
        try:
            prompt = input("User: ")
        except EOFError:
            break
        if prompt.strip().lower() in {"quit", "exit"}:
            break
        if not prompt.strip():
            continue
        answer = _generate(model, tokenizer, device, prompt, cfg)
        print(f"Agent: {answer}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="quin-llm: a small hand-written LLM")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tokenize", help="train the BPE tokenizer")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser("prepare-pretrain", help="tokenize pretraining corpora")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_prepare_pretrain)

    p = sub.add_parser("pretrain", help="train the base language model")
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=cmd_pretrain)

    p = sub.add_parser("prepare-sft", help="build token-level SFT samples")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default=None)
    p.set_defaults(func=cmd_prepare_sft)

    p = sub.add_parser("sft", help="fine-tune on instruction data")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True, help="pretrained checkpoint")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--resume", default=None)
    p.set_defaults(func=cmd_sft)

    for name, func in (("generate", cmd_generate), ("chat", cmd_chat)):
        p = sub.add_parser(name, help="run inference")
        p.add_argument("--ckpt", required=True)
        p.add_argument("--tokenizer", default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--max-new-tokens", type=int, default=200)
        p.add_argument("--temperature", type=float, default=0.8)
        p.add_argument("--top-k", type=int, default=40)
        p.add_argument("--top-p", type=float, default=0.95)
        p.add_argument("--repetition-penalty", type=float, default=1.1)
        if name == "generate":
            p.add_argument("--prompt", required=True)
        p.set_defaults(func=func)

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
