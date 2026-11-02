import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from config import ModelConfig, TrainConfig
from model.model import TinyModel
from trainer.tiny_trainer import Trainer


class _FixedDataset(Dataset):
    def __init__(self, vocab_size=300, block_size=8, size=8):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (
            torch.randint(0, self.vocab_size, (self.block_size,)),
            torch.randint(0, self.vocab_size, (self.block_size,)),
        )


class TestEarlyStopping(unittest.TestCase):
    def test_stops_after_patience(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_cfg = ModelConfig(
                vocab_size=300,
                block_size=8,
                n_layer=1,
                n_embd=32,
                n_head=4,
                dropout=0.0,
            )
            train_cfg = TrainConfig(
                out_dir=tmp,
                batch_size=2,
                grad_accum_steps=1,
                lr=1e-3,
                min_lr=1e-4,
                warmup_steps=0,
                max_steps=10,
                eval_interval=1,
                eval_batches=1,
                early_stop_patience=2,
                min_delta=0.0,
                save_interval=0,
                log_interval=1,
                amp=False,
                num_workers=0,
                seed=0,
                draw=False,
            )
            tokenizer = SimpleNamespace(vocab_size=300, pad_id=0, special_ids={})
            trainer = Trainer(TinyModel(model_cfg), tokenizer, train_cfg)
            losses = iter([1.0, 1.1, 1.2, 0.9])
            trainer.evaluate = lambda loader: next(losses)

            best = trainer.train_pretrain(_FixedDataset(), _FixedDataset())

            self.assertAlmostEqual(best, 1.0)
            final = torch.load(
                Path(tmp) / "final.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(final["step"], 3)
            self.assertEqual(final["evals_without_improvement"], 2)
            best_ckpt = torch.load(
                Path(tmp) / "best.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(best_ckpt["step"], 1)


if __name__ == "__main__":
    unittest.main()
