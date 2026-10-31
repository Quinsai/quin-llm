import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from pipeline.pretrain.dataset import PretrainDataset
from pipeline.sft.dataset import SFTDataset
from pipeline.sft.prepare import prepare_sft
from pipeline.sources import parse_sft_file
from pipeline.tokenizer.bpe_tokenizer import BPETokenizer


class TestSources(unittest.TestCase):
    def test_parse_sft_file_keeps_blank_lines(self):
        text = (
            "###User:\n"
            "Question one?\n"
            "###Agent:\n"
            "First paragraph.\n"
            "\n"
            "Second paragraph.\n"
            "\n"
            "###User:\n"
            "Question two?\n"
            "###Agent:\n"
            "Short answer.\n"
            "\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sft.txt"
            path.write_text(text, encoding="utf-8")
            samples = list(parse_sft_file(path))
        self.assertEqual(len(samples), 2)
        self.assertIn("\n\nSecond paragraph.", samples[0][1])
        self.assertEqual(samples[1][0], "Question two?")


class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer(vocab_size=300)
        cls.tokenizer.train(
            ["###User:\nhello\n###Agent:\nworld answer.\n" * 20],
            vocab_size=300,
            min_freq=1,
            max_words=5000,
            verbose=False,
        )

    def test_pretrain_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.bin"
            np.arange(100, dtype=np.uint16).tofile(path)
            ds = PretrainDataset(path, block_size=8, dtype="uint16", random_windows=False)
            self.assertEqual(len(ds), 12)
            x, y = ds[0]
            self.assertEqual(tuple(x.shape), (8,))
            self.assertEqual(tuple(y.shape), (8,))
            self.assertTrue(torch.equal(x[1:], y[:-1]))

            random_ds = PretrainDataset(path, block_size=8, dtype="uint16", random_windows=True)
            rx, ry = random_ds[3]
            self.assertEqual(tuple(rx.shape), (8,))
            self.assertEqual(tuple(ry.shape), (8,))

    def test_sft_masking_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "sft.txt"
            source.write_text(
                "###User:\nWhat is a cat?\n###Agent:\nA small animal.\n\n",
                encoding="utf-8",
            )
            cfg = {
                "out_dir": str(tmp / "out"),
                "tokenizer_path": "",
                "val_fraction": 0.0,
                "seed": 0,
                "source": {"type": "file", "path": str(source)},
            }
            prepare_sft(cfg, self.tokenizer, block_size=32)
            data = torch.load(tmp / "out" / "train.pt", weights_only=False)
            input_ids, labels = data["input_ids"], data["labels"]
            # One extra token is stored so the dataset can shift without loss.
            self.assertEqual(tuple(input_ids.shape), (1, 33))
            self.assertEqual(tuple(labels.shape), (1, 33))

            # Prompt tokens must be masked; at least one answer token must remain.
            masked = labels[0] == -100
            self.assertGreater(int(masked.sum()), 0)
            self.assertLess(int(masked.sum()), 33)
            self.assertTrue((input_ids[0][masked] >= 0).all())

            # Padding uses the pad id, and its labels are ignored.
            pad_positions = input_ids[0] == self.tokenizer.pad_id
            self.assertTrue(bool((labels[0][pad_positions] == -100).all()))

            # The dataset returns next-token targets, at most block_size long.
            x, y = SFTDataset(tmp / "out" / "train.pt")[0]
            self.assertEqual(tuple(x.shape), (32,))
            self.assertEqual(tuple(y.shape), (32,))
            self.assertTrue(torch.equal(x, input_ids[0][:-1].long()))
            self.assertTrue(torch.equal(y, labels[0][1:].long()))

            # The target immediately after the prompt is the first answer
            # token, not the answer token aligned with the same input position.
            first_answer = int(torch.nonzero(labels[0] != -100)[0])
            self.assertGreater(first_answer, 0)
            self.assertEqual(int(y[first_answer - 1]), int(input_ids[0][first_answer]))

            # With a larger stored width the dataset trims padding instead of
            # always returning block_size tokens.
            short_cfg = dict(cfg)
            short_cfg["out_dir"] = str(tmp / "short")
            prepare_sft(short_cfg, self.tokenizer, block_size=64)
            short_data = torch.load(tmp / "short" / "train.pt", weights_only=False)
            short_labels = short_data["labels"][0]
            valid_len = int(torch.nonzero(short_labels != -100)[-1]) + 1
            short_x, short_y = SFTDataset(tmp / "short" / "train.pt")[0]
            self.assertEqual(short_x.size(0), valid_len - 1)
            self.assertEqual(short_y.size(0), valid_len - 1)
            self.assertLess(short_x.size(0), 64)


if __name__ == "__main__":
    unittest.main()
