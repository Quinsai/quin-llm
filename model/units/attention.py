import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Uses PyTorch's fused ``scaled_dot_product_attention`` by default.  A manual
    implementation is kept for comparison/debugging (``use_sdpa=False``).
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float, use_sdpa: bool = True) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.use_sdpa = use_sdpa

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, seq_len, n_embd = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(n_embd, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~causal, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        return self.resid_dropout(self.proj(y))
