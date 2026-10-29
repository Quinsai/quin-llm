import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    """Pre-LN decoder block."""

    def __init__(self, n_embd: int, n_head: int, dropout: float, use_sdpa: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, use_sdpa=use_sdpa)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
