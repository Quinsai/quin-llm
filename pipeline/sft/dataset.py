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

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, index: int):
        # ``TinyModel.forward`` performs an unshifted cross-entropy on (x, y),
        # so the target at position i must be the token at input position i+1.
        # The prepared files store one extra token to make the shift lossless:
        # x has exactly block_size tokens and y[i] == labels[i + 1].
        input_ids = self.input_ids[index].long()
        labels = self.labels[index].long()
        return input_ids[:-1], labels[1:]


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
