"""SFT datasets with token-level prompt masking."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        data = torch.load(path, map_location="cpu", weights_only=False)
        # Stored as int32 to keep the files small; convert per item.
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]
        if self.input_ids.shape != self.labels.shape:
            raise ValueError("input_ids and labels must have the same shape")
        self.lengths = self._valid_lengths()

    def _valid_lengths(self) -> torch.Tensor:
        """Return the real prompt+answer length of every row.

        Prepared rows are padded to ``block_size + 1``.  The last non-ignored
        label is the final answer token (or EOS), so everything after it is
        padding and does not need to be fed to the model.
        """
        if self.labels.size(0) == 0:
            return torch.empty(0, dtype=torch.long)
        mask = self.labels != -100
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every SFT sample must contain at least one answer token")
        # Index of the first non-ignored label when read from the end.
        tail = mask.flip(1).to(torch.int8).argmax(dim=1)
        lengths = self.labels.size(1) - tail
        if int(lengths.min()) < 2:
            raise ValueError("SFT samples must contain at least a prompt and an answer token")
        return lengths

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int):
        # ``TinyModel.forward`` performs an unshifted cross-entropy on (x, y),
        # so the target at position i must be the token at input position i+1.
        # The prepared files store one extra token to make the shift lossless:
        # y[i] == labels[i + 1].
        length = int(self.lengths[index])
        return (
            self.input_ids[index, : length - 1].long(),
            self.labels[index, 1:length].long(),
        )


class ReplayDataset(Dataset):
    """Mix SFT samples with pretraining windows to reduce forgetting."""

    def __init__(self, sft_dataset: Dataset, replay_dataset: Dataset, replay_prob: float = 0.05):
        self.sft_dataset = sft_dataset
        self.replay_dataset = replay_dataset
        self.replay_prob = float(replay_prob)

    def seed_rng(self, seed: int) -> None:
        random.seed(seed)
        if hasattr(self.replay_dataset, "seed_rng"):
            self.replay_dataset.seed_rng(seed + 1)

    def __len__(self) -> int:
        return len(self.sft_dataset)

    def __getitem__(self, index: int):
        if self.replay_prob > 0 and random.random() < self.replay_prob:
            idx = random.randrange(len(self.replay_dataset))
            return self.replay_dataset[idx]
        return self.sft_dataset[index]
