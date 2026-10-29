import unittest

import torch

from config import ModelConfig
from model.model import TinyModel


class TestTinyModel(unittest.TestCase):
    def _config(self, use_sdpa=True):
        return ModelConfig(
            vocab_size=300,
            block_size=16,
            n_layer=2,
            n_embd=32,
            n_head=4,
            dropout=0.0,
            tie_weights=True,
            use_sdpa=use_sdpa,
        )

    def test_forward_shapes(self):
        model = TinyModel(self._config())
        x = torch.randint(0, 300, (2, 16))
        y = torch.randint(0, 300, (2, 16))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, 300))
        self.assertTrue(torch.isfinite(loss).item())

    def test_manual_attention_matches_shape(self):
        model = TinyModel(self._config(use_sdpa=False))
        x = torch.randint(0, 300, (2, 16))
        y = torch.randint(0, 300, (2, 16))
        logits, loss = model(x, y)
        self.assertEqual(tuple(logits.shape), (2, 16, 300))
        self.assertTrue(torch.isfinite(loss).item())

    def test_generate(self):
        model = TinyModel(self._config())
        model.train()
        x = torch.randint(0, 300, (1, 4))
        out = model.generate(x, max_new_tokens=8, temperature=0.8, top_k=10)
        self.assertEqual(tuple(out.shape), (1, 12))
        self.assertTrue(model.training, "generate should restore training mode")

    def test_checkpoint_roundtrip(self):
        model = TinyModel(self._config())
        x = torch.randint(0, 300, (1, 16))
        with torch.no_grad():
            before = model(x)
        clone = TinyModel(self._config())
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            after = clone(x)
        self.assertTrue(torch.allclose(before, after))

    def test_overfit_one_batch(self):
        torch.manual_seed(0)
        model = TinyModel(self._config())
        x = torch.randint(0, 300, (4, 16))
        y = torch.randint(0, 300, (4, 16))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        first = None
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            loss.backward()
            optimizer.step()
            if first is None:
                first = loss.item()
        self.assertLess(loss.item(), first * 0.8)
        self.assertLess(loss.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
