"""Memory-mapped pretraining dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


class PretrainDataset(Dataset):
    """Random (train) or fixed (val) windows from a flat token file.

    The file is a concatenation of documents, each wrapped in BOS/EOS.  A
    window of ``block_size + 1`` tokens gives inputs ``x`` and next-token
    targets ``y``.
    """

    def __init__(
        self,
        bin_path: str | Path,
        block_size: int,
        dtype: str | np.dtype = "uint16",
        random_windows: bool = True,
        seed: int = 42,
    ) -> None:
        self.bin_path = str(bin_path)
        self.block_size = int(block_size)
        self.data = np.memmap(self.bin_path, dtype=np.dtype(dtype), mode="r")
        if len(self.data) < self.block_size + 1:
            raise ValueError(
                f"{self.bin_path} has {len(self.data)} tokens, "
                f"need at least {self.block_size + 1}"
            )
        self.random_windows = random_windows
        self.num_windows = (len(self.data) - 1) // self.block_size
        self.rng = random.Random(seed)

    def seed_rng(self, seed: int) -> None:
        """Give each DataLoader worker its own random-window stream."""
        self.rng.seed(seed)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int):
        if self.random_windows:
            max_start = len(self.data) - self.block_size - 1
            start = 0 if max_start <= 0 else self.rng.randrange(0, max_start + 1)
        else:
            start = (int(idx) % self.num_windows) * self.block_size
        chunk = self.data[start : start + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def load_pretrain_meta(bin_dir: str | Path) -> dict:
    with open(Path(bin_dir) / "meta.json", "r", encoding="utf-8") as f:
        return json.load(f)
