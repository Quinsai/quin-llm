import json
import tempfile
import unittest
from pathlib import Path

from evaluation.freeze_datasets import build_manifest, verify_manifest
from evaluation.metrics import (
    alias_match,
    bootstrap_ci,
    check_constraint,
    exact_match,
    mcnemar,
    repetition_rate,
    score_prediction,
    summarize_records,
    token_f1,
)


class AnswerMetricTest(unittest.TestCase):
    def test_alias_match_accepts_short_answers_and_articles(self):
        self.assertTrue(alias_match("Paris", ["Paris"]))
        self.assertTrue(alias_match("The capital is Paris.", ["Paris"]))
        self.assertTrue(alias_match("the United States", ["United States"]))
        self.assertTrue(alias_match("Madrid.", ["Madrid", "Madrid city"]))
        self.assertFalse(alias_match("Lyon", ["Paris"]))
        self.assertFalse(
            alias_match(
                "Paris is the capital of France and it is a large city with "
                "many museums and parks and restaurants.",
                ["Paris"],
            )
        )

    def test_exact_match(self):
        self.assertTrue(exact_match("the Paris", ["Paris"]))
        self.assertFalse(exact_match("Paris, France", ["Paris"]))

    def test_token_f1(self):
        self.assertEqual(token_f1("the cat sat", "the cat sat"), 1.0)
        self.assertAlmostEqual(token_f1("the cat", "the cat sat"), 0.8)
        self.assertEqual(token_f1("dog", "cat"), 0.0)
        self.assertEqual(token_f1("", "cat"), 0.0)

    def test_repetition_rate_detects_loops(self):
        self.assertEqual(repetition_rate("one two three"), 0.0)
        looped = "the same four words " * 10
        self.assertGreater(repetition_rate(looped), 0.5)


class ConstraintTest(unittest.TestCase):
    def test_constraints(self):
        self.assertTrue(check_constraint("one two three", {"name": "max_words", "value": 3})[0])
        self.assertFalse(check_constraint("one two three", {"name": "max_words", "value": 2})[0])

        self.assertTrue(
            check_constraint("One. Two.", {"name": "exact_sentences", "value": 2})[0]
        )
        self.assertFalse(
            check_constraint("One. Two. Three.", {"name": "exact_sentences", "value": 2})[0]
        )

        self.assertTrue(
            check_constraint("First, add water.", {"name": "starts_with", "value": "First,"})[0]
        )
        self.assertFalse(
            check_constraint("Add water first.", {"name": "starts_with", "value": "First,"})[0]
        )

        bullets = "- alpha\n- beta\n- gamma"
        self.assertTrue(
            check_constraint(bullets, {"name": "bullet_count", "value": 3})[0]
        )
        self.assertFalse(
            check_constraint("- alpha\n- beta", {"name": "bullet_count", "value": 3})[0]
        )

        self.assertTrue(
            check_constraint("Water contains salt.", {"name": "includes_word", "value": "salt"})[0]
        )
        self.assertFalse(
            check_constraint("Water contains sugar.", {"name": "includes_word", "value": "salt"})[0]
        )

        self.assertTrue(check_constraint("all lowercase", {"name": "lowercase"})[0])
        self.assertFalse(check_constraint("Not lowercase", {"name": "lowercase"})[0])

        ok, detail = check_constraint("x", {"name": "unknown"})
        self.assertFalse(ok)
        self.assertIn("error", detail)


class ScorePredictionTest(unittest.TestCase):
    def test_alias_and_f1_checks(self):
        score, correct, _ = score_prediction(
            {"check": "alias_match", "reference": "Paris", "aliases": ["Paris"]},
            "Paris.",
        )
        self.assertEqual((score, correct), (1.0, True))

        score, correct, _ = score_prediction(
            {"check": "token_f1", "reference": "the cat sat"}, "the cat"
        )
        self.assertAlmostEqual(score, 0.8)
        self.assertTrue(correct)

    def test_json_schema_check(self):
        example = {
            "check": "json_schema",
            "schema": {"fields": {"name": "string", "age": "integer"}},
        }
        score, correct, detail = score_prediction(
            example, '{"name": "Ada", "age": 36}'
        )
        self.assertEqual((score, correct), (1.0, True))
        self.assertEqual(detail["errors"], [])

        score, correct, detail = score_prediction(
            example, '```json\n{"name": "Ada", "age": "old"}\n```'
        )
        self.assertEqual((score, correct), (0.0, False))
        self.assertTrue(detail["errors"])

        score, correct, detail = score_prediction(example, "not json")
        self.assertEqual((score, correct), (0.0, False))
        self.assertIn("error", detail)

    def test_set_match_reports_precision_recall(self):
        example = {
            "check": "set_match",
            "reference": ["14 March 2026", "15 March 2026"],
        }
        score, correct, detail = score_prediction(
            example, '["14 March 2026", "15 March 2026"]'
        )
        self.assertEqual((score, correct), (1.0, True))
        self.assertEqual(detail["precision"], 1.0)
        self.assertEqual(detail["recall"], 1.0)

        score, correct, detail = score_prediction(example, '["14 March 2026"]')
        self.assertAlmostEqual(score, 2 / 3)
        self.assertFalse(correct)
        self.assertAlmostEqual(detail["recall"], 0.5)

    def test_constraint_regex_keyword_nonempty(self):
        self.assertEqual(
            score_prediction(
                {"check": "constraint", "constraint": {"name": "max_words", "value": 2}},
                "two words",
            )[:2],
            (1.0, True),
        )
        self.assertEqual(
            score_prediction({"check": "regex", "pattern": r"^\d{3}$"}, "123")[:2],
            (1.0, True),
        )
        self.assertEqual(
            score_prediction({"check": "keyword", "keywords": ["salt", "water"]}, "water with salt")[:2],
            (1.0, True),
        )
        self.assertEqual(
            score_prediction({"check": "nonempty"}, "I'm sorry, I can't help.")[:2],
            (0.0, False),
        )

    def test_unscored_checks_return_none(self):
        score, correct, detail = score_prediction(
            {"check": "llm_judge", "rubric": "be helpful"}, "a story"
        )
        self.assertIsNone(score)
        self.assertIsNone(correct)
        self.assertTrue(detail["unscored"])
        self.assertTrue(detail["nonempty"])

    def test_unknown_check_raises(self):
        with self.assertRaises(ValueError):
            score_prediction({"check": "does_not_exist"}, "x")


class StatisticsTest(unittest.TestCase):
    def test_bootstrap_ci_is_deterministic(self):
        values = [0.0, 1.0, 1.0, 1.0, 0.5]
        first = bootstrap_ci(values, n_boot=200, seed=7)
        second = bootstrap_ci(values, n_boot=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertLessEqual(first[0], first[1])
        self.assertIsNone(bootstrap_ci([]))

    def test_mcnemar_exact(self):
        baseline = [True] * 8 + [False] * 2
        candidate = [False] * 8 + [True] * 2
        result = mcnemar(baseline, candidate)
        self.assertEqual(result["b"], 8)
        self.assertEqual(result["c"], 2)
        self.assertEqual(result["statistic"], 2.0)
        self.assertAlmostEqual(result["p_value"], 0.109375)

    def test_mcnemar_ignores_unscored(self):
        self.assertIsNone(mcnemar([None, None], [True, False]))
        self.assertEqual(
            mcnemar([True], [True]),
            {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0},
        )

    def test_summarize_records_groups_and_skips_unscored(self):
        records = [
            {"category": "fact_qa", "split": "seen", "score": 1.0, "correct": True},
            {"category": "fact_qa", "split": "seen", "score": 0.0, "correct": False},
            {"category": "retention", "split": "retention", "score": None, "correct": None},
        ]
        summary = summarize_records(records, n_boot=100)
        self.assertEqual(summary["overall"]["n"], 3)
        self.assertEqual(summary["overall"]["scored"], 2)
        self.assertEqual(summary["overall"]["unscored"], 1)
        self.assertEqual(summary["overall"]["accuracy"], 0.5)
        self.assertEqual(summary["by_category"]["fact_qa"]["n"], 2)
        self.assertIsNone(summary["by_category"]["retention"]["accuracy"])
        self.assertIn("fact_qa/seen", summary["by_category_split"])


class FreezeManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._write(
            "a.jsonl",
            [
                {"id": "a1", "category": "fact_qa", "split": "seen", "instruction": "Q1", "check": "alias_match"},
                {"id": "a2", "category": "fact_qa", "split": "seen", "instruction": "Q2", "check": "alias_match"},
            ],
        )
        self._write(
            "b.jsonl",
            [
                {"id": "b1", "category": "retention", "split": "retention", "instruction": "Q3", "check": "llm_judge"},
            ],
        )

    def _write(self, name, rows):
        with open(self.root / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    def test_build_manifest_counts_and_hashes(self):
        manifest = build_manifest(self.root)
        self.assertEqual(manifest["total_samples"], 3)
        self.assertEqual(len(manifest["files"]), 2)
        self.assertEqual(manifest["categories"], {"fact_qa": 2, "retention": 1})
        self.assertTrue(all(len(f["sha256"]) == 64 for f in manifest["files"]))

    def test_verify_detects_changes(self):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(
            json.dumps(build_manifest(self.root)), encoding="utf-8"
        )
        self.assertEqual(verify_manifest(self.root, manifest_path), [])

        self._write(
            "b.jsonl",
            [
                {"id": "b1", "category": "retention", "split": "retention", "instruction": "changed", "check": "llm_judge"},
            ],
        )
        problems = verify_manifest(self.root, manifest_path)
        self.assertTrue(any("hash mismatch" in p for p in problems))

    def test_duplicate_ids_are_rejected(self):
        self._write(
            "a.jsonl",
            [
                {"id": "same", "category": "fact_qa", "split": "seen", "instruction": "Q1", "check": "alias_match"},
            ],
        )
        self._write(
            "b.jsonl",
            [
                {"id": "same", "category": "retention", "split": "retention", "instruction": "Q2", "check": "llm_judge"},
            ],
        )
        with self.assertRaises(ValueError):
            build_manifest(self.root)

    def test_missing_required_field_is_rejected(self):
        self._write(
            "a.jsonl",
            [{"id": "a1", "category": "fact_qa", "split": "seen", "instruction": "Q1"}],
        )
        with self.assertRaises(ValueError):
            build_manifest(self.root)


if __name__ == "__main__":
    unittest.main()
