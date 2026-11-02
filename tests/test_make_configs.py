import json
import tempfile
import unittest
from pathlib import Path

from config import SFTConfig
from distillation.make_configs import run_make_configs

ROOT = Path(__file__).resolve().parents[1]


class MakeConfigsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.budget = self.root / "budget.json"
        self.budget.write_text(json.dumps({"max_steps": 1234}), encoding="utf-8")

    def test_generates_matched_controlled_configs(self):
        paths = run_make_configs(
            budget_path=str(self.budget),
            base_path=str(ROOT / "configs" / "sft_full.yaml"),
            out_dir=str(self.root / "generated"),
            run_id="pilot_v1",
            mixed_data="data/distill/sft/pilot_v1/train.parquet",
        )
        self.assertEqual(len(paths), 2)

        baseline = SFTConfig.from_yaml(paths[0])
        mixed = SFTConfig.from_yaml(paths[1])

        self.assertEqual(baseline.train.max_steps, 1234)
        self.assertEqual(mixed.train.max_steps, 1234)
        self.assertIsNone(baseline.train.early_stop_patience)
        self.assertIsNone(mixed.train.early_stop_patience)
        self.assertIsNone(baseline.train.epochs)
        self.assertIsNone(mixed.train.epochs)
        self.assertNotEqual(baseline.data.out_dir, mixed.data.out_dir)
        self.assertNotEqual(baseline.train.out_dir, mixed.train.out_dir)

        self.assertEqual(baseline.data.source["type"], "alpaca")
        self.assertEqual(
            mixed.data.source["paths"],
            "data/distill/sft/pilot_v1/train.parquet",
        )
        self.assertIn("sft_baseline_pilot_v1", paths[0].name)
        self.assertIn("sft_distill_pilot_v1", paths[1].name)

    def test_explicit_max_steps_does_not_need_budget(self):
        paths = run_make_configs(
            budget_path=str(self.root / "missing.json"),
            base_path=str(ROOT / "configs" / "sft_full.yaml"),
            out_dir=str(self.root / "generated"),
            run_id="smoke",
            mixed_data="data/distill/sft/smoke/train.parquet",
            max_steps=10,
        )
        self.assertEqual(SFTConfig.from_yaml(paths[0]).train.max_steps, 10)
        self.assertEqual(SFTConfig.from_yaml(paths[1]).train.max_steps, 10)

    def test_missing_budget_fails_with_guidance(self):
        with self.assertRaises(FileNotFoundError):
            run_make_configs(
                budget_path=str(self.root / "missing.json"),
                base_path=str(ROOT / "configs" / "sft_full.yaml"),
                out_dir=str(self.root / "generated"),
                run_id="pilot_v1",
                mixed_data="data/distill/sft/pilot_v1/train.parquet",
            )


if __name__ == "__main__":
    unittest.main()
