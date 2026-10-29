import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learned absolute position embeddings."""

    def __init__(self, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.block_size = block_size
        self.pos_emb = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        _, seq_len = x.shape
        if seq_len > self.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.block_size}")
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.pos_emb(pos)
