import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    def __init__(self, n_embd: int, vocab_size: int, bias: bool = False) -> None:
        super().__init__()
        self.lh = nn.Linear(n_embd, vocab_size, bias=bias)

    def forward(self, x, y=None):
        logits = self.lh(x)
        if y is None:
            return logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100,
        )
        return logits, loss
