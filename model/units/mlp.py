import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(n_embd, expansion * n_embd)
        self.act = nn.GELU()
        self.proj = nn.Linear(expansion * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(self.act(self.fc(x))))
