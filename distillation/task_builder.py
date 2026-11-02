"""Build a deterministic list of teacher tasks.

The task list is a file, not a live generator: generation, filtering, review,
and training all refer to the same ``task_id`` values.  Re-running with the
same config, seed, and seed files produces the same list.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from config import DistillGenerationConfig

from .prompts import (
    EXTRACTION_FIELDS,
    INSTRUCTION_CONSTRAINTS,
    JSON_SCHEMAS,
    PROMPT_VERSION,
    REASONING_TOPICS,
)
from .text_utils import canonical_json, iter_jsonl, normalize_text


DEFAULT_DIFFICULTIES = ("easy",)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(str(path)))


def load_entities(path: str | Path) -> list[dict[str, Any]]:
    entities = _read_jsonl(path)
    seen: set[str] = set()
    for item in entities:
        name = str(item.get("entity", "")).strip()
        if not name:
            raise ValueError(f"{path}: entity entry is missing 'entity'")
        key = normalize_text(name)
        if key in seen:
            raise ValueError(f"{path}: duplicate entity {name!r}")
        seen.add(key)
    return entities


def load_topics(path: str | Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    topics = []
    for item in rows:
        topic = str(item.get("topic", "")).strip()
        if topic:
            topics.append({"topic": topic})
    if not topics:
        raise ValueError(f"{path}: no topics found")
    return topics


def assert_disjoint_entities(train_path: str | Path, eval_path: str | Path) -> None:
    """Fail if an evaluation entity also appears in the training seeds."""
    train = {
        normalize_text(str(item.get("entity", "")))
        for item in load_entities(train_path)
    }
    eval_ = {
        normalize_text(str(item.get("entity", "")))
        for item in load_entities(eval_path)
    }
    overlap = sorted(train & eval_)
    if overlap:
        raise ValueError(
            "train/eval entity seeds overlap (evaluation contamination): "
            + ", ".join(overlap)
        )


def _difficulties(spec: dict[str, Any]) -> list[str]:
    raw = spec.get("difficulties", spec.get("difficulty", DEFAULT_DIFFICULTIES))
    if isinstance(raw, str):
        raw = [raw]
    values = [str(x) for x in raw if str(x).strip()]
    return values or list(DEFAULT_DIFFICULTIES)


def _seed_entities(spec: dict[str, Any], count: int, rng: random.Random) -> list[dict[str, Any]]:
    path = spec.get("seeds")
    if not path:
        raise ValueError("fact_qa needs a 'seeds' entity file")
    entities = load_entities(path)
    out: list[dict[str, Any]] = []
    for index in range(count):
        entity = entities[index % len(entities)]
        facts = entity.get("facts") or {}
        seed: dict[str, Any] = {
            "entity": entity["entity"],
            "type": entity.get("type", ""),
        }
        if entity.get("passage"):
            seed["passage"] = entity["passage"]
        if facts:
            relations = list(facts)
            relation = relations[rng.randrange(len(relations))]
            seed["relation"] = relation
            seed["expected"] = list(facts[relation])
        elif entity.get("relation"):
            seed["relation"] = entity["relation"]
        out.append(seed)
    return out


def _seed_topics(
    spec: dict[str, Any], count: int, rng: random.Random, fallback: list[str] | None = None
) -> list[str]:
    path = spec.get("seeds")
    if path:
        topics = [item["topic"] for item in load_topics(path)]
    else:
        topics = list(fallback or [])
    if not topics:
        raise ValueError("topic-based category needs a 'seeds' topic file")
    return [topics[rng.randrange(len(topics))] for _ in range(count)]


def _build_category(
    category: str, spec: dict[str, Any], rng: random.Random
) -> list[dict[str, Any]]:
    count = int(spec.get("count", 0) or 0)
    if count <= 0:
        return []
    difficulties = _difficulties(spec)
    tasks: list[dict[str, Any]] = []

    if category == "fact_qa":
        seeds = _seed_entities(spec, count, rng)
        for index, seed in enumerate(seeds):
            tasks.append(
                {
                    "task_id": f"{category}-{index:06d}",
                    "category": category,
                    "difficulty": difficulties[index % len(difficulties)],
                    "seed": seed,
                    "source": Path(str(spec.get("seeds"))).name,
                }
            )
        return tasks

    if category == "format_json":
        topics = _seed_topics(spec, count, rng, fallback=None)
        for index, topic in enumerate(topics):
            schema = JSON_SCHEMAS[rng.randrange(len(JSON_SCHEMAS))]
            tasks.append(
                {
                    "task_id": f"{category}-{index:06d}",
                    "category": category,
                    "difficulty": difficulties[index % len(difficulties)],
                    "seed": {"topic": topic, "schema": schema},
                    "source": Path(str(spec.get("seeds"))).name,
                }
            )
        return tasks

    if category == "extraction":
        topics = _seed_topics(spec, count, rng, fallback=None)
        for index, topic in enumerate(topics):
            field = EXTRACTION_FIELDS[rng.randrange(len(EXTRACTION_FIELDS))]
            tasks.append(
                {
                    "task_id": f"{category}-{index:06d}",
                    "category": category,
                    "difficulty": difficulties[index % len(difficulties)],
                    "seed": {
                        "topic": topic,
                        "field": field["name"],
                        "field_description": field["description"],
                    },
                    "source": Path(str(spec.get("seeds"))).name,
                }
            )
        return tasks

    if category == "instruction":
        topics = _seed_topics(spec, count, rng, fallback=None)
        for index, topic in enumerate(topics):
            constraint = INSTRUCTION_CONSTRAINTS[
                rng.randrange(len(INSTRUCTION_CONSTRAINTS))
            ]
            tasks.append(
                {
                    "task_id": f"{category}-{index:06d}",
                    "category": category,
                    "difficulty": difficulties[index % len(difficulties)],
                    "seed": {"topic": topic, "constraint": constraint},
                    "source": Path(str(spec.get("seeds"))).name,
                }
            )
        return tasks

    if category == "reasoning":
        topics = _seed_topics(spec, count, rng, fallback=REASONING_TOPICS)
        for index, topic in enumerate(topics):
            tasks.append(
                {
                    "task_id": f"{category}-{index:06d}",
                    "category": category,
                    "difficulty": difficulties[index % len(difficulties)],
                    "seed": {"topic": topic},
                    "source": (
                        Path(str(spec.get("seeds"))).name
                        if spec.get("seeds")
                        else "builtin_reasoning_topics"
                    ),
                }
            )
        return tasks

    raise ValueError(f"unknown category: {category!r}")


def build_tasks(cfg: DistillGenerationConfig) -> list[dict[str, Any]]:
    if not cfg.categories:
        raise ValueError("distillation config has no categories")

    tasks: list[dict[str, Any]] = []
    for category, raw_spec in cfg.categories.items():
        spec = raw_spec if isinstance(raw_spec, dict) else {"count": int(raw_spec)}
        # Seed the RNG per category so changing one category's count does not
        # reshuffle the others.
        rng = random.Random(f"{cfg.seed}:{category}")
        tasks.extend(_build_category(category, spec, rng))

    for task in tasks:
        task["prompt_version"] = cfg.prompt_version
        task["task_hash"] = hashlib.sha256(
            canonical_json(task).encode("utf-8")
        ).hexdigest()
    return tasks


def write_tasks(tasks: list[dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")


def task_manifest(tasks: list[dict[str, Any]], cfg: DistillGenerationConfig) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    return {
        "run_id": cfg.run_id,
        "prompt_version": cfg.prompt_version,
        "seed": cfg.seed,
        "total": len(tasks),
        "counts": counts,
        "prompt_template_version": PROMPT_VERSION,
    }


def run_task_builder(cfg: DistillGenerationConfig) -> Path:
    tasks = build_tasks(cfg)
    out = Path(cfg.tasks_dir) / f"{cfg.run_id}.jsonl"
    write_tasks(tasks, out)
    manifest = task_manifest(tasks, cfg)
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"tasks: {len(tasks):,} -> {out}")
    print("  " + ", ".join(f"{k}={v:,}" for k, v in sorted(manifest["counts"].items())))
    return out


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    tasks = _read_jsonl(path)
    seen: set[str] = set()
    for task in tasks:
        task_id = task.get("task_id")
        if not task_id:
            raise ValueError(f"{path}: task is missing task_id")
        if task_id in seen:
            raise ValueError(f"{path}: duplicate task_id {task_id!r}")
        seen.add(task_id)
    return tasks
