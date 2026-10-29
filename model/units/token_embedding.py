import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)

    def forward(self, x):
        return self.token_emb(x)
