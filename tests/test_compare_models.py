import json
import tempfile
import unittest
from pathlib import Path

from evaluation.compare_models import run_compare


def record(checkpoint, *, correct=None, score=None):
    return {
        "id": "retention-1",
        "category": "retention",
        "split": "retention",
        "instruction": "Write a short story about a fox.",
        "prediction": "A fox walked through the snow.",
        "reference": "",
        "correct": correct,
        "score": score,
        "checkpoint": checkpoint,
    }


class CompareModelsTest(unittest.TestCase):
    def test_unscored_retention_rows_do_not_shadow_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.jsonl"
            candidate = root / "candidate.jsonl"
            report = root / "report.md"
            baseline.write_text(
                json.dumps(record("baseline-ckpt")) + "\n", encoding="utf-8"
            )
            candidate.write_text(
                json.dumps(record("mixed-ckpt")) + "\n", encoding="utf-8"
            )

            metrics = run_compare(
                str(baseline), str(candidate), str(report), n_boot=10
            )

            self.assertEqual(metrics["paired"], 1)
            self.assertEqual(metrics["unscored"], 1)
            text = report.read_text(encoding="utf-8")
            self.assertIn("baseline-ckpt", text)
            self.assertIn("mixed-ckpt", text)


if __name__ == "__main__":
    unittest.main()
