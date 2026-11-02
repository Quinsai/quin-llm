import json
import tempfile
import unittest
from pathlib import Path

from evaluation.contamination_check import run_check

ROOT = Path(__file__).resolve().parents[1]


class ContaminationCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_eval(self, rows):
        path = self.root / "eval.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def write_train(self, rows):
        path = self.root / "train.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    def test_seen_entity_mentions_are_not_entity_hits(self):
        eval_path = self.write_eval(
            [
                {
                    "id": "seen",
                    "split": "seen_entity_seen_template",
                    "instruction": "What is the capital of France?",
                    "reference": "Paris",
                    "metadata": {"entity": "France"},
                }
            ]
        )
        train = self.write_train(
            [
                {
                    "instruction": "Describe France in one sentence.",
                    "output": "France is a country in Europe.",
                }
            ]
        )
        stats = run_check(
            str(train),
            str(eval_path),
            entity_seeds=str(ROOT / "distillation" / "seeds" / "eval_entities.jsonl"),
        )
        self.assertEqual(stats["entity_hits"], 0)
        self.assertEqual(stats["exact_instruction_hits"], 0)
        self.assertEqual(stats["total_hits"], 0)

    def test_held_out_entity_mentions_are_hits(self):
        eval_path = self.write_eval(
            [
                {
                    "id": "unseen",
                    "split": "unseen_entity",
                    "instruction": "What is the capital of Spain?",
                    "reference": "Madrid",
                    "metadata": {"entity": "Spain"},
                }
            ]
        )
        train = self.write_train(
            [
                {
                    "instruction": "Describe Spain in one sentence.",
                    "output": "Spain is a country in Europe.",
                }
            ]
        )
        stats = run_check(
            str(train),
            str(eval_path),
            # No seed file: the hit must come from the eval row's held-out split.
            entity_seeds=str(self.root / "missing-entities.jsonl"),
        )
        self.assertEqual(stats["entity_hits"], 1)

    def test_external_floor_entity_is_not_an_entity_hit(self):
        # External-floor subjects include generic words ("gold", "1984"), so
        # that split is guarded by exact/near matching rather than entity hits.
        eval_path = self.write_eval(
            [
                {
                    "id": "floor",
                    "split": "external_floor",
                    "instruction": "What is the capital of Peru?",
                    "reference": "Lima",
                    "metadata": {"entity": "Peru"},
                }
            ]
        )
        train = self.write_train(
            [
                {
                    "instruction": "Describe Peru in one sentence.",
                    "output": "Peru is a country in South America.",
                }
            ]
        )
        stats = run_check(
            str(train),
            str(eval_path),
            entity_seeds=str(self.root / "missing-entities.jsonl"),
        )
        self.assertEqual(stats["entity_hits"], 0)

    def test_exact_instruction_is_always_a_hit(self):
        eval_path = self.write_eval(
            [
                {
                    "id": "seen",
                    "split": "seen_entity_seen_template",
                    "instruction": "What is the capital of France?",
                    "reference": "Paris",
                    "metadata": {"entity": "France"},
                }
            ]
        )
        train = self.write_train(
            [
                {
                    "instruction": "What is the capital of France?",
                    "output": "Paris.",
                }
            ]
        )
        stats = run_check(
            str(train),
            str(eval_path),
            entity_seeds=str(ROOT / "distillation" / "seeds" / "eval_entities.jsonl"),
        )
        self.assertEqual(stats["exact_instruction_hits"], 1)
        self.assertEqual(stats["entity_hits"], 0)


if __name__ == "__main__":
    unittest.main()
