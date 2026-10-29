"""Train a byte-level BPE tokenizer from a YAML config."""

from __future__ import annotations

from pathlib import Path

from config import TokenizerConfig
from data_pipeline.tokenizer.bpe_tokenizer import BPETokenizer, iter_texts_from_sources


def train_tokenizer(config: TokenizerConfig | str | Path) -> BPETokenizer:
    cfg = config if isinstance(config, TokenizerConfig) else TokenizerConfig.from_yaml(config)
    if not cfg.sources:
        raise ValueError("tokenizer config has no sources")

    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    print(f"training tokenizer: vocab_size={tokenizer.target_vocab_size}")
    tokenizer.train(
        iter_texts_from_sources(cfg.sources, max_chars=cfg.max_chars),
        vocab_size=cfg.vocab_size,
        min_freq=cfg.min_freq,
        max_words=cfg.max_words,
        max_chars=cfg.max_chars,
    )
    tokenizer.save(cfg.out_path)
    print(f"saved tokenizer to {cfg.out_path} (vocab_size={tokenizer.vocab_size})")
    return tokenizer
