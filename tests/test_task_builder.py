import json
import tempfile
import unittest
from pathlib import Path

from config import DistillGenerationConfig
from distillation.prompts import (
    EXTRACTION_FIELDS,
    INSTRUCTION_CONSTRAINTS,
    JSON_SCHEMAS,
    build_messages,
    schema_for_example,
)
from distillation.task_builder import (
    assert_disjoint_entities,
    build_tasks,
    load_entities,
    run_task_builder,
)

ROOT = Path(__file__).resolve().parents[1]


class TaskBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def config(self, **overrides):
        values = {
            "run_id": "test",
            "out_dir": str(self.root / "raw"),
            "tasks_dir": str(self.root / "tasks"),
            "prompt_version": "v1",
            "seed": 42,
            "categories": {
                "fact_qa": {
                    "count": 20,
                    "seeds": str(ROOT / "distillation" / "seeds" / "train_entities.jsonl"),
                    "difficulty": ["easy", "medium"],
                },
                "format_json": {
                    "count": 6,
                    "seeds": str(ROOT / "distillation" / "seeds" / "topics.jsonl"),
                },
                "extraction": {
                    "count": 6,
                    "seeds": str(ROOT / "distillation" / "seeds" / "topics.jsonl"),
                },
                "instruction": {
                    "count": 6,
                    "seeds": str(ROOT / "distillation" / "seeds" / "topics.jsonl"),
                },
                "reasoning": {"count": 6},
            },
        }
        values.update(overrides)
        return DistillGenerationConfig(**values)

    def test_tasks_are_deterministic_and_unique(self):
        first = build_tasks(self.config())
        second = build_tasks(self.config())
        self.assertEqual(first, second)
        ids = [task["task_id"] for task in first]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(first), 44)

        for task in first:
            self.assertTrue(task["task_hash"])
            self.assertIn("seed", task)
            messages = build_messages(task)
            self.assertEqual([m["role"] for m in messages], ["system", "user"])
            self.assertTrue(all(m["content"].strip() for m in messages))

    def test_fact_tasks_use_seed_facts(self):
        tasks = build_tasks(self.config())
        facts = [t for t in tasks if t["category"] == "fact_qa"]
        for task in facts:
            self.assertTrue(task["seed"].get("entity"))
            self.assertTrue(task["seed"].get("expected"))
            self.assertTrue(task["seed"].get("relation"))

    def test_json_and_extraction_seeds_are_complete(self):
        tasks = build_tasks(self.config())
        for task in tasks:
            if task["category"] == "format_json":
                self.assertIn(task["seed"]["schema"], JSON_SCHEMAS)
                self.assertTrue(task["seed"]["topic"])
            elif task["category"] == "extraction":
                self.assertTrue(task["seed"]["field"])
                self.assertTrue(task["seed"]["field_description"])
            elif task["category"] == "instruction":
                self.assertIn(task["seed"]["constraint"], INSTRUCTION_CONSTRAINTS)
            elif task["category"] == "reasoning":
                self.assertTrue(task["seed"]["topic"])

    def test_write_tasks_and_manifest(self):
        path = run_task_builder(self.config(run_id="smoke"))
        self.assertTrue(path.exists())
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 44)
        manifest = json.loads(
            path.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["run_id"], "smoke")
        self.assertEqual(manifest["total"], 44)
        self.assertEqual(manifest["counts"]["fact_qa"], 20)

    def test_entity_splits_are_disjoint(self):
        assert_disjoint_entities(
            ROOT / "distillation" / "seeds" / "train_entities.jsonl",
            ROOT / "distillation" / "seeds" / "eval_entities.jsonl",
        )
        self.assertEqual(
            len(load_entities(ROOT / "distillation" / "seeds" / "train_entities.jsonl")),
            246,
        )

    def test_overlapping_entity_splits_fail(self):
        bad = self.root / "eval_bad.jsonl"
        bad.write_text(
            json.dumps({"entity": "France", "type": "country"}) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            assert_disjoint_entities(
                ROOT / "distillation" / "seeds" / "train_entities.jsonl", bad
            )

    def test_fact_tasks_use_distinct_facts(self):
        tasks = build_tasks(self.config())
        facts = [t for t in tasks if t["category"] == "fact_qa"]
        keys = [
            (
                task["seed"]["entity"],
                task["seed"]["relation"],
                tuple(task["seed"].get("expected") or []),
            )
            for task in facts
        ]
        self.assertEqual(len(keys), 20)
        self.assertEqual(len(keys), len(set(keys)))

    def test_reasoning_tasks_use_distinct_topics(self):
        tasks = build_tasks(self.config())
        topics = [
            task["seed"]["topic"]
            for task in tasks
            if task["category"] == "reasoning"
        ]
        self.assertEqual(len(topics), 6)
        self.assertEqual(len(topics), len(set(topics)))

    def test_schema_validation_accepts_nested_arrays(self):
        schema = JSON_SCHEMAS[3]  # product, includes array of strings
        errors = schema_for_example(
            {
                "name": "Mouse",
                "price": 24.99,
                "in_stock": True,
                "tags": ["electronics", "accessories"],
            },
            schema,
        )
        self.assertEqual(errors, [])

        errors = schema_for_example(
            {
                "name": "Mouse",
                "price": "cheap",
                "in_stock": "yes",
                "tags": "electronics",
            },
            schema,
        )
        self.assertEqual(len(errors), 3)


if __name__ == "__main__":
    unittest.main()
