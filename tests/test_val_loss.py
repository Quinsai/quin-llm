import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from configs.schema import ModelConfig
from evaluation.val_loss import evaluate_val_loss
from model.model import TinyModel


class ValLossTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data_dir = self.root / "pretrain"
        self.data_dir.mkdir()

        model_cfg = ModelConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_embd=16,
            n_head=2,
            dropout=0.0,
        )
        torch.manual_seed(0)
        model = TinyModel(model_cfg)
        self.ckpt = self.root / "ckpt.pt"
        torch.save(
            {
                "model_config": model_cfg.to_dict(),
                "model_state": model.state_dict(),
                "step": 7,
            },
            self.ckpt,
        )

        tokens = np.random.default_rng(0).integers(
            0, 64, size=40, dtype=np.uint16
        )
        tokens.tofile(self.data_dir / "val.bin")
        (self.data_dir / "meta.json").write_text(
            json.dumps({"dtype": "uint16", "val_tokens": int(len(tokens))}),
            encoding="utf-8",
        )

    def test_evaluates_finite_loss(self):
        result = evaluate_val_loss(
            str(self.ckpt),
            data_dir=str(self.data_dir),
            batch_size=2,
            batches=2,
            device="cpu",
            amp=False,
        )
        self.assertEqual(result["checkpoint_step"], 7)
        self.assertEqual(result["block_size"], 8)
        self.assertEqual(result["batches"], 2)
        self.assertEqual(result["windows"], 4)
        self.assertGreater(result["val_loss"], 0.0)
        self.assertGreater(result["perplexity"], 1.0)
        self.assertFalse(result["amp"])

    def test_missing_val_file_is_an_error(self):
        (self.data_dir / "val.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            evaluate_val_loss(
                str(self.ckpt),
                data_dir=str(self.data_dir),
                device="cpu",
                amp=False,
            )


if __name__ == "__main__":
    unittest.main()
