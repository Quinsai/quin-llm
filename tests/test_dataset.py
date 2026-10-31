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
            meta = prepare_sft(cfg, self.tokenizer, block_size=64)
            data = torch.load(tmp / "out" / "train.pt", weights_only=False)
            input_ids, labels = data["input_ids"], data["labels"]
            # One extra token is stored so the dataset can shift without loss.
            self.assertEqual(tuple(input_ids.shape), (1, 65))
            self.assertEqual(tuple(labels.shape), (1, 65))

            # Prompt tokens must be masked; at least one answer token must remain.
            masked = labels[0] == -100
            self.assertGreater(int(masked.sum()), 0)
            self.assertLess(int(masked.sum()), 65)
            self.assertTrue((input_ids[0][masked] >= 0).all())

            # Padding uses the pad id, and its labels are ignored.
            pad_positions = input_ids[0] == self.tokenizer.pad_id
            self.assertTrue(bool(pad_positions.any()))
            self.assertTrue(bool((labels[0][pad_positions] == -100).all()))

            # The dataset trims padding and returns next-token targets that are
            # at most block_size tokens long.
            valid_len = int(torch.nonzero(labels[0] != -100)[-1]) + 1
            answer_tokens = int((labels[0] != -100).sum())
            self.assertLess(valid_len, 65)
            self.assertEqual(meta["train_total_tokens"], valid_len)
            self.assertEqual(meta["train_answer_tokens"], answer_tokens)
            self.assertEqual(meta["val_total_tokens"], 0)
            self.assertEqual(meta["val_answer_tokens"], 0)
            self.assertEqual(meta["total_tokens"], valid_len)
            self.assertEqual(meta["answer_tokens"], answer_tokens)
            self.assertAlmostEqual(meta["avg_total_tokens"], float(valid_len))
            self.assertEqual(meta["max_total_tokens"], valid_len)
            self.assertAlmostEqual(meta["avg_answer_tokens"], float(answer_tokens))
            self.assertEqual(meta["max_answer_tokens"], answer_tokens)
            x, y = SFTDataset(tmp / "out" / "train.pt")[0]
            self.assertEqual(x.size(0), valid_len - 1)
            self.assertEqual(y.size(0), valid_len - 1)
            self.assertTrue(torch.equal(x, input_ids[0][: valid_len - 1].long()))
            self.assertTrue(torch.equal(y, labels[0][1:valid_len].long()))

            # The target immediately after the prompt is the first answer
            # token, not the answer token aligned with the same input position.
            first_answer = int(torch.nonzero(labels[0] != -100)[0])
            self.assertGreater(first_answer, 0)
            self.assertEqual(int(y[first_answer - 1]), int(input_ids[0][first_answer]))

    def test_sft_skips_overlong_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "sft.txt"
            source.write_text(
                "###User:\nWhat is a cat?\n###Agent:\nA small animal.\n\n"
                "###User:\nTell me a long story.\n###Agent:\n"
                + ("word " * 500)
                + "\n\n",
                encoding="utf-8",
            )
            cfg = {
                "out_dir": str(tmp / "out"),
                "tokenizer_path": "",
                "val_fraction": 0.0,
                "seed": 0,
                "source": {"type": "file", "path": str(source)},
            }
            meta = prepare_sft(cfg, self.tokenizer, block_size=64)
            self.assertEqual(meta["train_samples"], 1)
            self.assertEqual(meta["skipped"], 1)
            self.assertEqual(meta["overlong_samples"], 1)
            data = torch.load(tmp / "out" / "train.pt", weights_only=False)
            self.assertEqual(data["input_ids"].shape[0], 1)
            labels = data["labels"][0]
            valid_len = int(torch.nonzero(labels != -100)[-1]) + 1
            self.assertEqual(meta["train_total_tokens"], valid_len)
            self.assertEqual(meta["answer_tokens"], int((labels != -100).sum()))
            self.assertEqual(meta["total_tokens"], meta["train_total_tokens"])


if __name__ == "__main__":
    unittest.main()
