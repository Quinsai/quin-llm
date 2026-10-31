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

import torch

from config import (
    InferenceConfig,
    PreparePretrainConfig,
    PrepareSFTConfig,
    PretrainConfig,
    SFTConfig,
    TokenizerConfig,
)
from pipeline.common import build_model_from_checkpoint, load_tokenizer, resolve_device
from pipeline.pretrain.train import run_prepare_pretrain, run_pretrain
from pipeline.sft.prepare import PROMPT_TEMPLATE
from pipeline.sft.train import run_prepare_sft, run_sft
from pipeline.tokenizer.train import train_tokenizer
from trainer.tiny_trainer import Trainer


# ----------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------
def cmd_tokenize(args) -> None:
    train_tokenizer(TokenizerConfig.from_yaml(args.config))


# ----------------------------------------------------------------------
# pretraining
# ----------------------------------------------------------------------
def cmd_prepare_pretrain(args) -> None:
    run_prepare_pretrain(PreparePretrainConfig.from_yaml(args.config))


def cmd_pretrain(args) -> None:
    run_pretrain(PretrainConfig.from_yaml(args.config), resume_from=args.resume)


# ----------------------------------------------------------------------
# supervised fine-tuning
# ----------------------------------------------------------------------
def cmd_prepare_sft(args) -> None:
    run_prepare_sft(PrepareSFTConfig.from_yaml(args.config), ckpt=args.ckpt)


def cmd_sft(args) -> None:
    run_sft(
        SFTConfig.from_yaml(args.config),
        ckpt=args.ckpt,
        epochs=args.epochs,
        resume_from=args.resume,
    )


# ----------------------------------------------------------------------
# inference
# ----------------------------------------------------------------------
def _load_for_inference(cfg: InferenceConfig):
    device = resolve_device(cfg.device)
    checkpoint = Trainer.load_checkpoint(cfg.ckpt, map_location="cpu")
    tokenizer_path = cfg.tokenizer or checkpoint.get("tokenizer_path")
    if not tokenizer_path:
        raise ValueError("no tokenizer path in checkpoint; pass --tokenizer")
    tokenizer = load_tokenizer(tokenizer_path)
    model = build_model_from_checkpoint(checkpoint, device)
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
