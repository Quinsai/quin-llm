import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from config import DistillGenerationConfig
from distillation import data_generator
from distillation.data_generator import _interleave_by_category, run_generation


class FakeTeacherClient:
    model = "fake-teacher"
    base_url = "http://fake.test/v1"

    def __init__(self, cfg):
        self.cfg = cfg

    def generation_params(self):
        return {"temperature": 0.8, "top_p": 0.95, "max_tokens": 256}

    def request_id(self, identity, version):
        return f"req-{identity}"

    def describe(self):
        return {
            "base_url": self.base_url,
            "model": self.model,
            "params": self.generation_params(),
        }

    def chat(self, messages):
        return {"choices": [{"message": {"content": "{}"}}]}

    def extract_content(self, data):
        return '{"instruction": "Q", "output": "A"}'

    def finish_reason(self, data):
        return "stop"

    def usage(self, data):
        return {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


class DataGeneratorTest(unittest.TestCase):
    def test_interleave_by_category_balances_limited_runs(self):
        items = []
        for category, count in (("fact_qa", 3), ("format_json", 2), ("extraction", 1)):
            for index in range(count):
                items.append(
                    (
                        {"task_id": f"{category}-{index}", "category": category},
                        [],
                        f"id-{category}-{index}",
                    )
                )
        ordered = _interleave_by_category(items)
        self.assertEqual(
            [item[0]["task_id"] for item in ordered],
            ["fact_qa-0", "format_json-0", "extraction-0",
             "fact_qa-1", "format_json-1", "fact_qa-2"],
        )

    def test_usage_sums_resumed_and_new_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "raw"
            out_dir.mkdir()
            tasks_dir = root / "tasks"
            tasks_dir.mkdir()
            teacher_yaml = root / "teacher.yaml"
            teacher_yaml.write_text(
                "base_url: http://fake.test/v1\n"
                "model: fake-teacher\n"
                "api_key: fake-key\n"
                "concurrency: 1\n",
                encoding="utf-8",
            )
            raw_path = out_dir / "run.jsonl"
            raw_path.write_text(
                json.dumps(
                    {
                        "request_id": "old-request",
                        "status": "ok",
                        "content": "{}",
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 4,
                            "total_tokens": 7,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task = {
                "task_id": "fact-1",
                "category": "fact_qa",
                "difficulty": "easy",
                "seed": {
                    "entity": "France",
                    "type": "country",
                    "relation": "capital",
                    "expected": ["Paris"],
                },
                "source": "test",
                "prompt_version": "v1",
            }
            cfg = DistillGenerationConfig(
                run_id="run",
                teacher_config=str(teacher_yaml),
                out_dir=str(out_dir),
                tasks_dir=str(tasks_dir),
            )

            with mock.patch.object(
                data_generator, "run_task_builder", return_value=tasks_dir / "run.jsonl"
            ), mock.patch.object(
                data_generator, "load_tasks", return_value=[task]
            ), mock.patch.object(
                data_generator, "TeacherClient", FakeTeacherClient
            ):
                run_generation(cfg, limit=1)

            stats = json.loads((out_dir / "run.stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["usage_this_run"]["prompt_tokens"], 5)
            self.assertEqual(stats["usage_total"]["prompt_tokens"], 8)
            self.assertEqual(stats["usage_total"]["completion_tokens"], 11)
            self.assertEqual(stats["usage_total"]["total_tokens"], 19)


if __name__ == "__main__":
    unittest.main()
