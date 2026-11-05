import csv
import json
import tempfile
import unittest
from pathlib import Path

from config import DistillFilterConfig
from distillation.data_filter import filter_records
from distillation.prompts import build_user_prompt


def raw_record(
    task_id,
    category,
    instruction,
    output,
    *,
    seed=None,
    status="ok",
    finish_reason="stop",
):
    return {
        "task_id": task_id,
        "category": category,
        "difficulty": "easy",
        "seed": seed or {},
        "status": status,
        "finish_reason": finish_reason,
        "model": "qwen-test",
        "prompt_version": "v1",
        "request_id": f"req-{task_id}",
        "seed_source": "test",
        "content": json.dumps({"instruction": instruction, "output": output}),
    }


class FilterTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = self.root / "raw.jsonl"
        self.out = self.root / "filtered.jsonl"
        self.rejected = self.root / "rejected.jsonl"
        self.stats_path = self.root / "stats.json"
        self.review = self.root / "review.csv"
        self.eval_dir = self.root / "eval"
        self.eval_dir.mkdir()

    def write_raw(self, records):
        with open(self.raw, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def config(self, **overrides):
        values = {
            "tokenizer_path": str(self.root / "missing-tokenizer.pkl"),
            "eval_dir": str(self.eval_dir),
            "eval_entity_seeds": str(self.root / "missing-entities.jsonl"),
            "min_review_samples": 1,
            "min_rejected_review": 1,
            "review_fraction": 1.0,
            "rejected_review_fraction": 1.0,
        }
        values.update(overrides)
        return DistillFilterConfig(**values)

    def run_filter(self, records, **overrides):
        self.write_raw(records)
        cfg = self.config(**overrides)
        return filter_records(
            self.raw,
            self.out,
            cfg,
            rejected_path=self.rejected,
            stats_path=self.stats_path,
            review_path=self.review,
        )

    def kept(self):
        return [
            json.loads(line)
            for line in self.out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def rejected_rows(self):
        return [
            json.loads(line)
            for line in self.rejected.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class CategoryFilterTest(FilterTestBase):
    def test_valid_fact_is_kept(self):
        self.run_filter(
            [
                raw_record(
                    "fact-1",
                    "fact_qa",
                    "What is the capital of France?",
                    "The capital of France is Paris.",
                    seed={"entity": "France", "expected": ["Paris"]},
                )
            ]
        )
        rows = self.kept()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "teacher")
        self.assertEqual(rows[0]["task_id"], "fact-1")
        self.assertEqual(rows[0]["category"], "fact_qa")

    def test_fact_mismatch_is_rejected(self):
        self.run_filter(
            [
                raw_record(
                    "fact-1",
                    "fact_qa",
                    "What is the capital of France?",
                    "The capital of France is Lyon.",
                    seed={"entity": "France", "expected": ["Paris"]},
                )
            ]
        )
        rows = self.rejected_rows()
        self.assertIn("fact_mismatch", rows[0]["reasons"])

    def test_fact_answer_in_instruction_is_rejected(self):
        self.run_filter(
            [
                raw_record(
                    "fact-1",
                    "fact_qa",
                    "Is Paris the capital of France?",
                    "Yes, Paris is the capital.",
                    seed={"entity": "France", "expected": ["Paris"]},
                )
            ]
        )
        self.assertIn("answer_in_instruction", self.rejected_rows()[0]["reasons"])

    def test_json_schema_pass_and_fail(self):
        schema = {"fields": {"name": "string", "age": "integer"}}
        self.run_filter(
            [
                raw_record(
                    "json-ok",
                    "format_json",
                    "Return a JSON object with name and age.",
                    '{"name": "Ada", "age": 36}',
                    seed={"schema": schema},
                ),
                raw_record(
                    "json-bad",
                    "format_json",
                    "Return a JSON object with name and age.",
                    '{"name": "Ada", "age": "old"}',
                    seed={"schema": schema},
                ),
            ]
        )
        self.assertEqual([r["task_id"] for r in self.kept()], ["json-ok"])
        reasons = self.rejected_rows()[0]["reasons"]
        self.assertIn("schema_mismatch", reasons)

    def test_extraction_must_be_a_single_value_in_the_instruction(self):
        self.run_filter(
            [
                raw_record(
                    "extract-ok",
                    "extraction",
                    "Extract the date from: The meeting is on 14 March 2026.",
                    '["14 March 2026"]',
                ),
                raw_record(
                    "extract-missing",
                    "extraction",
                    "Extract the date from: The meeting is on 14 March 2026.",
                    '["16 March 2026"]',
                ),
                raw_record(
                    "extract-two",
                    "extraction",
                    "Extract the dates from: Meetings are on 14 March 2026 and 15 March 2026.",
                    '["14 March 2026", "15 March 2026"]',
                ),
            ]
        )
        self.assertEqual([r["task_id"] for r in self.kept()], ["extract-ok"])
        by_id = {r["task_id"]: r for r in self.rejected_rows()}
        self.assertIn("extraction_not_in_instruction", by_id["extract-missing"]["reasons"])
        self.assertIn("extraction_count", by_id["extract-two"]["reasons"])

    def test_instruction_constraints(self):
        constraint = {"name": "max_words", "value": 5}
        self.run_filter(
            [
                raw_record(
                    "instr-ok",
                    "instruction",
                    "Describe tea briefly.",
                    "Tea is a warm drink.",
                    seed={"constraint": constraint},
                ),
                raw_record(
                    "instr-bad",
                    "instruction",
                    "Describe tea in five words.",
                    "Tea is a very pleasant warm drink.",
                    seed={"constraint": constraint},
                ),
            ]
        )
        self.assertEqual([r["task_id"] for r in self.kept()], ["instr-ok"])
        self.assertIn(
            "constraint_max_words", self.rejected_rows()[0]["reasons"]
        )

    def test_reasoning_rejects_chain_of_thought(self):
        self.run_filter(
            [
                raw_record(
                    "reason-bad",
                    "reasoning",
                    "Why is the sky blue?",
                    "Step 1: sunlight enters the atmosphere. Step 2: blue light scatters more.",
                )
            ]
        )
        self.assertIn("chain_of_thought", self.rejected_rows()[0]["reasons"])

    def test_prompt_echo_is_rejected(self):
        seed = {"entity": "France", "expected": ["Paris"]}
        prompt = build_user_prompt(
            {"category": "fact_qa", "difficulty": "easy", "seed": seed}
        )
        self.run_filter(
            [
                raw_record(
                    "fact-echo",
                    "fact_qa",
                    prompt,
                    "The capital of France is Paris.",
                    seed=seed,
                )
            ]
        )
        reasons = self.rejected_rows()[0]["reasons"]
        self.assertIn("prompt_echo", reasons)

    def test_declared_word_count_constraints(self):
        self.run_filter(
            [
                raw_record(
                    "instr-ok",
                    "instruction",
                    "Describe tea in exactly 5 words.",
                    "Tea is a warm drink.",
                ),
                raw_record(
                    "instr-exact-bad",
                    "instruction",
                    "Describe coffee in exactly 5 words.",
                    "Coffee is a very warm drink.",
                ),
                raw_record(
                    "instr-max-bad",
                    "instruction",
                    "Describe milk in at most 5 words.",
                    "Milk is a very cold drink.",
                ),
            ]
        )
        by_id = {r["task_id"]: r["reasons"] for r in self.rejected_rows()}
        self.assertIn("constraint_declared_words", by_id["instr-exact-bad"])
        self.assertIn("constraint_declared_words", by_id["instr-max-bad"])
        self.assertNotIn("instr-ok", by_id)

    def test_finish_reason_not_stop_is_rejected(self):
        self.run_filter(
            [
                raw_record(
                    "fact-truncated",
                    "fact_qa",
                    "What is the capital of France?",
                    "The capital of France is",
                    seed={"entity": "France", "expected": ["Paris"]},
                    finish_reason="length",
                )
            ]
        )
        reasons = self.rejected_rows()[0]["reasons"]
        self.assertIn("finish_reason_length", reasons)

    def test_refusal_is_rejected(self):
        self.run_filter(
            [
                raw_record(
                    "fact-refusal",
                    "fact_qa",
                    "What is the capital of France?",
                    "I'm sorry, I can't answer that.",
                    seed={"entity": "France", "expected": ["Paris"]},
                )
            ]
        )
        self.assertIn("refusal", self.rejected_rows()[0]["reasons"])

    def test_api_error_is_rejected(self):
        self.run_filter(
            [
                raw_record(
                    "fact-error",
                    "fact_qa",
                    "What is the capital of France?",
                    "Paris.",
                    seed={"entity": "France", "expected": ["Paris"]},
                    status="error",
                )
            ]
        )
        self.assertIn("api_error", self.rejected_rows()[0]["reasons"])


class DedupAndContaminationTest(FilterTestBase):
    def test_duplicate_instruction_is_rejected(self):
        record = raw_record(
            "fact-1",
            "fact_qa",
            "What is the capital of France?",
            "The capital of France is Paris.",
            seed={"entity": "France", "expected": ["Paris"]},
        )
        duplicate = dict(record)
        duplicate["task_id"] = "fact-2"
        self.run_filter([record, duplicate])

        self.assertEqual(len(self.kept()), 1)
        self.assertIn("duplicate_instruction", self.rejected_rows()[0]["reasons"])

    def test_entity_cap(self):
        records = [
            raw_record(
                "fact-1",
                "fact_qa",
                "What is the capital of France?",
                "The capital of France is Paris.",
                seed={"entity": "France", "expected": ["Paris"]},
            ),
            raw_record(
                "fact-2",
                "fact_qa",
                "Which city is France's capital?",
                "Paris is the capital of France.",
                seed={"entity": "France", "expected": ["Paris"]},
            ),
        ]
        self.run_filter(records, max_per_entity=1)
        self.assertEqual(len(self.kept()), 1)
        self.assertIn("entity_cap", self.rejected_rows()[0]["reasons"])

    def test_exact_eval_instruction_is_rejected(self):
        eval_file = self.eval_dir / "eval.jsonl"
        eval_file.write_text(
            json.dumps(
                {
                    "id": "eval-1",
                    "category": "fact_qa",
                    "split": "seen",
                    "instruction": "What is the capital of France?",
                    "reference": "Paris",
                    "check": "alias_match",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.run_filter(
            [
                raw_record(
                    "fact-1",
                    "fact_qa",
                    "What is the capital of France?",
                    "The capital of France is Paris.",
                    seed={"entity": "France", "expected": ["Paris"]},
                )
            ]
        )
        reasons = self.rejected_rows()[0]["reasons"]
        self.assertIn("eval_exact_instruction", reasons)

    def test_held_out_entity_from_eval_metadata_is_rejected(self):
        eval_file = self.eval_dir / "eval.jsonl"
        eval_file.write_text(
            json.dumps(
                {
                    "id": "eval-1",
                    "category": "fact_qa",
                    "split": "unseen_entity",
                    "instruction": "Which city is the capital of Spain?",
                    "reference": "Madrid",
                    "check": "alias_match",
                    "metadata": {"entity": "Spain"},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.run_filter(
            [
                raw_record(
                    "fact-1",
                    "fact_qa",
                    "Describe the capital city of Spain in one sentence.",
                    "Madrid is the capital of Spain.",
                    seed={"entity": "Spain", "expected": ["Madrid"]},
                )
            ]
        )
        reasons = self.rejected_rows()[0]["reasons"]
        self.assertIn("eval_entity_mentioned", reasons)

    def test_held_out_entity_is_rejected(self):
        entity_file = self.root / "eval_entities.jsonl"
        entity_file.write_text(
            json.dumps({"entity": "Spain", "type": "country"}) + "\n",
            encoding="utf-8",
        )
        self.run_filter(
            [
                raw_record(
                    "fact-1",
                    "fact_qa",
                    "What is the capital of Spain?",
                    "The capital of Spain is Madrid.",
                    seed={"entity": "Spain", "expected": ["Madrid"]},
                )
            ],
            eval_entity_seeds=str(entity_file),
        )
        reasons = self.rejected_rows()[0]["reasons"]
        self.assertIn("eval_entity_mentioned", reasons)


class ArtifactTest(FilterTestBase):
    def test_stats_and_review_csv_are_written(self):
        stats = self.run_filter(
            [
                raw_record(
                    "fact-ok",
                    "fact_qa",
                    "What is the capital of France?",
                    "The capital of France is Paris.",
                    seed={"entity": "France", "expected": ["Paris"]},
                ),
                raw_record(
                    "fact-bad",
                    "fact_qa",
                    "What is the capital of France?",
                    "The capital of France is Lyon.",
                    seed={"entity": "France", "expected": ["Paris"]},
                ),
            ]
        )
        self.assertEqual(stats["raw_records"], 2)
        self.assertEqual(stats["kept"], 1)
        self.assertEqual(stats["rejected"], 1)
        self.assertIn("fact_mismatch", stats["reasons"])
        self.assertTrue(self.stats_path.exists())

        with open(self.review, "r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("model_verdict", rows[0])
        self.assertIn("human_verdict", rows[0])
        self.assertEqual(rows[0]["model_verdict"], "")
        self.assertEqual(rows[0]["human_verdict"], "")

    def test_malformed_raw_line_is_skipped(self):
        self.write_raw([])
        with open(self.raw, "w", encoding="utf-8") as f:
            f.write("{not json}\n")
            f.write(
                json.dumps(
                    raw_record(
                        "fact-1",
                        "fact_qa",
                        "What is the capital of France?",
                        "The capital of France is Paris.",
                        seed={"entity": "France", "expected": ["Paris"]},
                    )
                )
                + "\n"
            )
        cfg = self.config()
        stats = filter_records(
            self.raw,
            self.out,
            cfg,
            rejected_path=self.rejected,
            stats_path=self.stats_path,
            review_path=self.review,
        )
        self.assertEqual(stats["raw_records"], 1)
        self.assertEqual(stats["kept"], 1)


if __name__ == "__main__":
    unittest.main()
