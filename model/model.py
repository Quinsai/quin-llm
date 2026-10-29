"""Decoder-only Transformer language model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig

from .units.linear_head import LinearHead
from .units.position_embedding import PositionalEmbedding
from .units.token_embedding import TokenEmbedding
from .units.transformer_block import TransformerBlock


class TinyModel(nn.Module):
    def __init__(self, config: ModelConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = ModelConfig.from_dict(config)
        self.config = config

        self.token_embd = TokenEmbedding(config.vocab_size, config.n_embd)
        self.position_embd = PositionalEmbedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    n_embd=config.n_embd,
                    n_head=config.n_head,
                    dropout=config.dropout,
                    use_sdpa=config.use_sdpa,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.output_head = LinearHead(config.n_embd, config.vocab_size)

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual output projections.
        residual_std = 0.02 / math.sqrt(2 * config.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.proj.weight, mean=0.0, std=residual_std)

        self._tie_weights()

    def _tie_weights(self) -> None:
        if self.config.tie_weights:
            self.output_head.lh.weight = self.token_embd.token_emb.weight

    def _apply(self, fn, recurse: bool = True):
        # Moving a module to a new device/dtype replaces Parameter objects, which
        # would silently break weight tying.  Re-tie after every conversion.
        super()._apply(fn, recurse=recurse)
        self._tie_weights()
        return self

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embd.token_emb.weight.numel()
            n -= self.position_embd.pos_emb.weight.numel()
        return n

    def forward(self, x, y=None):
        _, seq_len = x.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
            )
        h = self.token_embd(x) + self.position_embd(x)
        h = self.drop(h)
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        return self.output_head(h, y)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample from the model.

        ``idx`` may contain a batch; generation stops when every sequence has
        emitted EOS (or after ``max_new_tokens``).
        """
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size :]
                logits = self(idx_cond)[:, -1, :].float()

                if repetition_penalty != 1.0:
                    for b in range(idx.size(0)):
                        tokens = torch.unique(idx[b])
                        scores = logits[b, tokens]
                        scores = torch.where(
                            scores > 0,
                            scores / repetition_penalty,
                            scores * repetition_penalty,
                        )
                        logits[b, tokens] = scores

                if temperature <= 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k is not None and top_k > 0:
                        k = min(int(top_k), logits.size(-1))
                        top_values, _ = torch.topk(logits, k, dim=-1)
                        logits = logits.masked_fill(
                            logits < top_values[:, -1].unsqueeze(-1), float("-inf")
                        )
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=True, dim=-1
                        )
                        probs = F.softmax(sorted_logits, dim=-1)
                        cumulative = probs.cumsum(dim=-1)
                        remove = cumulative - probs > top_p
                        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                        logits = torch.full_like(logits, float("-inf")).scatter(
                            1, sorted_indices, sorted_logits
                        )
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                idx = torch.cat((idx, next_token), dim=1)
                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break
        finally:
            self.train(was_training)
        return idx
