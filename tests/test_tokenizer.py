import tempfile
import unittest
from pathlib import Path

from data_pipeline.tokenizer.bpe_tokenizer import BPETokenizer


class TestBPETokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BPETokenizer(vocab_size=320)
        text = (
            "hello world, this is a small language model. "
            "the model learns from text and predicts the next token. "
        )
        cls.tokenizer.train(
            [text * 20],
            vocab_size=320,
            min_freq=1,
            max_words=5000,
            verbose=False,
        )

    def test_roundtrip_ascii(self):
        text = "Hello, world! This is a test."
        ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_roundtrip_unicode(self):
        text = "你好，世界! 😀 Café"
        ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        self.assertEqual(self.tokenizer.decode(ids), text)

    def test_special_tokens(self):
        ids = self.tokenizer.encode("hello", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], self.tokenizer.bos_id)
        self.assertEqual(ids[-1], self.tokenizer.eos_id)
        self.assertEqual(self.tokenizer.decode(ids), "hello")
        self.assertLess(self.tokenizer.pad_id, self.tokenizer.vocab_size)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.pkl"
            self.tokenizer.save(path)
            loaded = BPETokenizer.load(path)
            text = "the model learns from text"
            self.assertEqual(
                self.tokenizer.encode(text),
                loaded.encode(text),
            )
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded.decode(loaded.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
