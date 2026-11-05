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
    EXTRACTION_PROMPT_STYLES,
    FACT_PROMPT_STYLES,
    INSTRUCTION_CONSTRAINTS,
    INSTRUCTION_PROMPT_STYLES,
    JSON_PROMPT_STYLES,
    JSON_SCHEMAS,
    PROMPT_VERSION,
    REASONING_PROMPT_STYLES,
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
        if not topic:
            continue
        entry: dict[str, Any] = {"topic": topic}
        schemas = item.get("schemas")
        if isinstance(schemas, list):
            entry["schemas"] = [str(x).strip() for x in schemas if str(x).strip()]
        topics.append(entry)
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


def _topic_entries(
    spec: dict[str, Any], fallback: list[str] | None = None
) -> list[dict[str, Any]]:
    path = spec.get("seeds")
    if path:
        return load_topics(path)
    topics = [{"topic": topic} for topic in (fallback or [])]
    if not topics:
        raise ValueError("topic-based category needs a 'seeds' topic file")
    return topics


def _entity_fact_seeds(spec: dict[str, Any]) -> list[dict[str, Any]]:
    path = spec.get("seeds")
    if not path:
        raise ValueError("fact_qa needs a 'seeds' entity file")
    out: list[dict[str, Any]] = []
    for entity in load_entities(path):
        base: dict[str, Any] = {
            "entity": entity["entity"],
            "type": entity.get("type", ""),
        }
        passage = str(entity.get("passage") or "").strip()
        if passage:
            out.append({**base, "passage": passage})
            continue
        facts = entity.get("facts") or {}
        if facts:
            for relation, values in facts.items():
                expected = list(values) if isinstance(values, list) else [values]
                out.append({**base, "relation": relation, "expected": expected})
        elif entity.get("relation"):
            out.append({**base, "relation": entity["relation"]})
        else:
            out.append(base)
    if not out:
        raise ValueError(f"{path}: no usable entity facts")
    return out


def _format_items(spec: dict[str, Any]) -> list[dict[str, Any]]:
    schemas_by_name = {schema["name"]: schema for schema in JSON_SCHEMAS}
    items: list[dict[str, Any]] = []
    for entry in _topic_entries(spec):
        names = entry.get("schemas") or [schema["name"] for schema in JSON_SCHEMAS]
        for name in names:
            schema = schemas_by_name.get(name)
            if schema is None:
                raise ValueError(
                    f"topic {entry['topic']!r} references unknown schema {name!r}"
                )
            items.append({"topic": entry["topic"], "schema": schema})
    return items


def _take_tasks(
    items: list[dict[str, Any]],
    count: int,
    difficulties: list[str],
    style_count: int,
    category: str,
    rng: random.Random,
) -> list[tuple[dict[str, Any], str, int]]:
    """Pick tasks, preferring unique content before reusing a seed item.

    The teacher often writes the same question for the same fact even when the
    meta-prompt is reworded, so the first pass uses every content item once.
    Styles and difficulties are only extra capacity after that.
    """
    if not items:
        raise ValueError(f"{category}: no seed items")
    order = list(range(len(items)))
    rng.shuffle(order)
    selected: list[tuple[dict[str, Any], str, int]] = []
    used: set[tuple[int, str, int]] = set()

    def add(index: int, difficulty: str, style: int) -> bool:
        key = (index, difficulty, style)
        if key in used:
            return False
        used.add(key)
        selected.append((items[index], difficulty, style))
        return True

    # One task per distinct fact/topic/pair, with the least verbose style.
    for position, index in enumerate(order):
        if len(selected) >= count:
            break
        add(index, difficulties[position % len(difficulties)], 0)

    # Reuse content under other prompt styles before repeating combinations.
    for style in range(1, max(1, style_count)):
        if len(selected) >= count:
            break
        for position, index in enumerate(order):
            if len(selected) >= count:
                break
            add(index, difficulties[position % len(difficulties)], style)

    # Fill in the remaining difficulty/style combinations.
    if len(selected) < count:
        for style in range(max(1, style_count)):
            for index in order:
                for difficulty in difficulties:
                    if len(selected) >= count:
                        break
                    add(index, difficulty, style)
                if len(selected) >= count:
                    break
            if len(selected) >= count:
                break

    if len(selected) < count:
        print(
            f"warning: {category}: {count:,} tasks requested but only "
            f"{len(selected):,} unique (seed, style, difficulty) combinations; "
            "repeating content"
        )
        position = 0
        while len(selected) < count:
            index = order[position % len(order)]
            difficulty = difficulties[position % len(difficulties)]
            style = (position // len(order)) % max(1, style_count)
            selected.append((items[index], difficulty, style))
            position += 1
    return selected


def _build_category(
    category: str, spec: dict[str, Any], rng: random.Random
) -> list[dict[str, Any]]:
    count = int(spec.get("count", 0) or 0)
    if count <= 0:
        return []
    difficulties = _difficulties(spec)
    seeds_path = spec.get("seeds")

    if category == "fact_qa":
        items = _entity_fact_seeds(spec)
        selected = _take_tasks(
            items, count, difficulties, len(FACT_PROMPT_STYLES), category, rng
        )
        definitions = [
            (
                {**item, "style": style},
                Path(str(seeds_path)).name,
            )
            for item, _, style in selected
        ]
    elif category == "format_json":
        items = _format_items(spec)
        selected = _take_tasks(
            items, count, difficulties, len(JSON_PROMPT_STYLES), category, rng
        )
        definitions = [
            (
                {
                    "topic": item["topic"],
                    "schema": item["schema"],
                    "style": style,
                },
                Path(str(seeds_path)).name,
            )
            for item, _, style in selected
        ]
    elif category == "extraction":
        items = [
            {
                "topic": entry["topic"],
                "field": field["name"],
                "field_description": field["description"],
            }
            for entry in _topic_entries(spec)
            for field in EXTRACTION_FIELDS
        ]
        selected = _take_tasks(
            items, count, difficulties, len(EXTRACTION_PROMPT_STYLES), category, rng
        )
        definitions = [
            ({**item, "style": style}, Path(str(seeds_path)).name)
            for item, _, style in selected
        ]
    elif category == "instruction":
        items = [
            {"topic": entry["topic"], "constraint": constraint}
            for entry in _topic_entries(spec)
            for constraint in INSTRUCTION_CONSTRAINTS
        ]
        selected = _take_tasks(
            items, count, difficulties, len(INSTRUCTION_PROMPT_STYLES), category, rng
        )
        definitions = [
            ({**item, "style": style}, Path(str(seeds_path)).name)
            for item, _, style in selected
        ]
    elif category == "reasoning":
        items = [
            {"topic": entry["topic"]}
            for entry in _topic_entries(spec, fallback=REASONING_TOPICS)
        ]
        selected = _take_tasks(
            items, count, difficulties, len(REASONING_PROMPT_STYLES), category, rng
        )
        source = (
            Path(str(seeds_path)).name if seeds_path else "builtin_reasoning_topics"
        )
        definitions = [({**item, "style": style}, source) for item, _, style in selected]
    else:
        raise ValueError(f"unknown category: {category!r}")

    tasks: list[dict[str, Any]] = []
    for index, (seed, source) in enumerate(definitions):
        tasks.append(
            {
                "task_id": f"{category}-{index:06d}",
                "category": category,
                "difficulty": selected[index][1],
                "seed": seed,
                "source": source,
            }
        )
    return tasks


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
