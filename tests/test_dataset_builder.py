import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from config import DatasetBuildConfig
from distillation.dataset_builder import run_build_dataset


class DatasetBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.filtered = self.root / "filtered.jsonl"
        self.alpaca = self.root / "alpaca.jsonl"
        self.out_dir = self.root / "mix"
        self._write_teacher(rows=4, words=10)
        self._write_alpaca(rows=20, words=2)

    def _write_teacher(self, rows, words):
        with open(self.filtered, "w", encoding="utf-8") as f:
            for index in range(rows):
                f.write(
                    json.dumps(
                        {
                            "task_id": f"teacher-{index}",
                            "instruction": f"Teacher instruction {index}?",
                            "input": "",
                            "output": " ".join(
                                ["answer"] * words
                            )
                            + " .",
                            "category": "fact_qa",
                            "difficulty": "easy",
                            "source": "teacher",
                        }
                    )
                    + "\n"
                )

    def _write_alpaca(self, rows, words):
        with open(self.alpaca, "w", encoding="utf-8") as f:
            for index in range(rows):
                f.write(
                    json.dumps(
                        {
                            "instruction": f"Alpaca instruction {index}",
                            "input": "",
                            "output": " ".join(["short"] * words),
                        }
                    )
                    + "\n"
                )

    def config(self, **overrides):
        values = {
            "filtered_path": str(self.filtered),
            "alpaca_paths": str(self.alpaca),
            "tokenizer_path": str(self.root / "missing-tokenizer.pkl"),
            "baseline_budget": str(self.root / "missing-budget.json"),
            "out_dir": str(self.out_dir),
            "teacher_ratio": 0.5,
            "seed": 13,
        }
        values.update(overrides)
        return DatasetBuildConfig(**values)

    def read_mix(self):
        return pd.read_parquet(self.out_dir / "train.parquet")

    def test_mixes_by_answer_token_mass(self):
        manifest = run_build_dataset(self.config())
        frame = self.read_mix()

        self.assertIn("instruction", frame.columns)
        self.assertIn("output", frame.columns)
        self.assertIn("source", frame.columns)
        self.assertEqual(len(frame), manifest["mix"]["samples"]["total"])
        counts = frame["source"].value_counts().to_dict()
        self.assertEqual(counts.get("teacher"), manifest["mix"]["samples"]["teacher"])
        self.assertEqual(counts.get("alpaca"), manifest["mix"]["samples"]["alpaca"])
        self.assertGreater(manifest["mix"]["samples"]["teacher"], 0)
        self.assertGreater(manifest["mix"]["samples"]["alpaca"], 0)
        ratio = manifest["mix"]["actual_teacher_ratio"]
        self.assertAlmostEqual(ratio, 0.5, delta=0.06)
        self.assertTrue((out_dir := self.out_dir / "manifest.json").exists())
        stored = json.loads(out_dir.read_text(encoding="utf-8"))
        self.assertEqual(stored["mix"]["requested_teacher_ratio"], 0.5)
        self.assertIsNotNone(stored["filtered"]["sha256"])

    def test_target_samples_caps_row_count(self):
        manifest = run_build_dataset(self.config(target_samples=10))
        self.assertEqual(len(self.read_mix()), 10)
        self.assertEqual(manifest["mix"]["samples"]["total"], 10)

    def test_teacher_only(self):
        manifest = run_build_dataset(self.config(teacher_ratio=1.0))
        frame = self.read_mix()
        self.assertEqual(set(frame["source"]), {"teacher"})
        self.assertEqual(manifest["mix"]["actual_teacher_ratio"], 1.0)

    def test_alpaca_only(self):
        manifest = run_build_dataset(self.config(teacher_ratio=0.0))
        frame = self.read_mix()
        self.assertEqual(set(frame["source"]), {"alpaca"})
        self.assertEqual(manifest["mix"]["actual_teacher_ratio"], 0.0)

    def test_write_alpaca_control(self):
        manifest = run_build_dataset(self.config(write_alpaca_control=True))
        control = pd.read_parquet(self.out_dir / "alpaca_control.parquet")
        self.assertEqual(len(control), manifest["outputs"]["alpaca_control_rows"])
        self.assertEqual(len(control), manifest["mix"]["samples"]["total"])

    def test_missing_filtered_file_is_an_error(self):
        cfg = self.config(filtered_path=str(self.root / "nope.jsonl"))
        with self.assertRaises(FileNotFoundError):
            run_build_dataset(cfg)

    def test_none_values_are_cleaned(self):
        with open(self.alpaca, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "instruction": "Do something",
                        "input": None,
                        "output": "Done.",
                    }
                )
                + "\n"
            )
        run_build_dataset(self.config(teacher_ratio=0.0))
        frame = self.read_mix()
        self.assertNotIn("None", frame["instruction"].iloc[0])
        self.assertEqual(frame["input"].iloc[0], "")


if __name__ == "__main__":
    unittest.main()
