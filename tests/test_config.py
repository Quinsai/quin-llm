import unittest
from pathlib import Path

from config import (
    InferenceConfig,
    ModelConfig,
    PreparePretrainConfig,
    PrepareSFTConfig,
    PretrainConfig,
    SFTConfig,
    TokenizerConfig,
    TrainConfig,
)

ROOT = Path(__file__).resolve().parents[1]


class TestStageConfigs(unittest.TestCase):
    def test_mini_pretrain_config(self):
        cfg = PretrainConfig.from_yaml(ROOT / "configs" / "pretrain_mini.yaml")
        self.assertEqual(cfg.model.vocab_size, 1024)
        self.assertEqual(cfg.model.n_layer, 2)
        self.assertEqual(cfg.data.out_dir, "processed_data/pretrain_mini")
        self.assertFalse(cfg.train.amp)
        self.assertEqual(cfg.train.max_steps, 100)

    def test_mini_prepare_configs(self):
        tok = TokenizerConfig.from_yaml(ROOT / "configs" / "tokenizer_mini.yaml")
        self.assertEqual(tok.out_path, "processed_data/tokenizer_mini/tokenizer.pkl")
        self.assertEqual(len(tok.sources), 2)

        prep = PreparePretrainConfig.from_yaml(ROOT / "configs" / "pretrain_mini.yaml")
        self.assertEqual(prep.data.out_dir, "processed_data/pretrain_mini")
        self.assertEqual(len(prep.data.sources), 1)

        prep_sft = PrepareSFTConfig.from_yaml(ROOT / "configs" / "sft_mini.yaml")
        self.assertEqual(prep_sft.data.out_dir, "processed_data/sft_mini")
        self.assertEqual(prep_sft.data.block_size, 128)

    def test_mini_sft_config(self):
        cfg = SFTConfig.from_yaml(ROOT / "configs" / "sft_mini.yaml")
        self.assertEqual(cfg.data.source["type"], "file")
        self.assertEqual(cfg.train.max_steps, 60)
        self.assertFalse(cfg.train.amp)

    def test_low_level_reexports(self):
        self.assertEqual(ModelConfig().block_size, 512)
        self.assertEqual(TrainConfig().batch_size, 64)
        self.assertTrue(hasattr(InferenceConfig(), "max_new_tokens"))


if __name__ == "__main__":
    unittest.main()
