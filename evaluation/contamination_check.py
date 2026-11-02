"""Check training data against the frozen evaluation sets.

A non-zero exit means the training data overlaps evaluation data and should not
be trained on until the offending rows are removed.  Exact/near-duplicate
instructions and exact pairs are always checked.  Entity mentions are checked
only for *held-out* evaluation entities (the ``unseen_entity`` split plus
``eval_entities.jsonl``): the seen-entity splits intentionally mention trained
entities, so treating those as contamination would make the check impossible
to pass.  ``external_floor`` items are covered by exact/near matching only,
because several of their subjects are generic words ("gold", "1984").

Example:
    python -m evaluation.contamination_check \
        --train data/distill/sft/pilot_v1/train.parquet \
        --eval evaluation/datasets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from distillation.text_utils import (
    NearDuplicateIndex,
    iter_jsonl,
    normalize_answer,
    normalize_text,
)

from .metrics import _whole_phrase_in  # reuse the same phrase rule as scoring

# Entity mentions are only hard contamination for splits whose entities were
# deliberately held out of training.  Seen-entity eval rows share entities with
# the seed files, so entity matching there would flag legitimate training data.
# External-floor items use exact/near matching only: their subjects include
# generic words ("gold", "1984") that would cause false positives.
HELD_OUT_ENTITY_SPLITS = {"unseen_entity"}


def _train_rows(path: str | Path) -> Iterable[dict[str, str]]:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pandas as pd

        frame = pd.read_parquet(path)
        for row in frame.to_dict(orient="records"):
            instruction = str(row.get("instruction") or "").strip()
            extra = str(row.get("input") or "").strip()
            output = str(row.get("output") or "").strip()
            yield {
                "instruction": instruction + ("\n" + extra if extra else ""),
                "output": output,
            }
        return
    if path.suffix.lower() == ".jsonl":
        for row in iter_jsonl(str(path)):
            instruction = str(row.get("instruction") or "").strip()
            extra = str(row.get("input") or "").strip()
            output = str(row.get("output") or "").strip()
            yield {
                "instruction": instruction + ("\n" + extra if extra else ""),
                "output": output,
            }
        return
    raise ValueError(f"unsupported training file: {path}")


def _eval_records(eval_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(eval_dir)
    files = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    records = []
    for path in files:
        for record in iter_jsonl(str(path)):
            record = dict(record)
            record["_file"] = str(path)
            records.append(record)
    return records


def _pair_key(instruction: str, output: str) -> str:
    return normalize_text(instruction) + "\n" + normalize_answer(output)


def run_check(
    train_path: str,
    eval_path: str,
    *,
    threshold: float = 0.85,
    entity_seeds: str | None = "distillation/seeds/eval_entities.jsonl",
    max_examples: int = 5,
) -> dict[str, Any]:
    eval_records = _eval_records(eval_path)
    if not eval_records:
        raise ValueError(f"no evaluation records under {eval_path}")

    exact_instructions: set[str] = set()
    exact_pairs: set[str] = set()
    entities: set[str] = set()
    near = NearDuplicateIndex(threshold=threshold)
    for index, record in enumerate(eval_records):
        instruction = str(record.get("instruction") or "").strip()
        reference = str(
            record.get("reference")
            or record.get("output")
            or record.get("answer")
            or ""
        ).strip()
        sample_id = str(record.get("id") or f"eval-{index}")
        if instruction:
            exact_instructions.add(normalize_text(instruction))
            near.add(instruction, sample_id)
            if reference:
                exact_pairs.add(_pair_key(instruction, reference))
        metadata = record.get("metadata") or {}
        entity = metadata.get("entity") or record.get("entity")
        split = str(record.get("split", ""))
        if entity and split in HELD_OUT_ENTITY_SPLITS:
            entities.add(str(entity))

    if entity_seeds:
        seed_path = Path(entity_seeds)
        if seed_path.exists():
            for item in iter_jsonl(str(seed_path)):
                entity = str(item.get("entity", "")).strip()
                if entity:
                    entities.add(entity)

    stats = {
        "train_path": train_path,
        "eval_path": eval_path,
        "threshold": threshold,
        "train_rows": 0,
        "eval_rows": len(eval_records),
        "eval_entities": len(entities),
        "exact_instruction_hits": 0,
        "near_instruction_hits": 0,
        "exact_pair_hits": 0,
        "entity_hits": 0,
        "examples": {
            "exact_instruction": [],
            "near_instruction": [],
            "exact_pair": [],
            "entity": [],
        },
    }

    for row in _train_rows(train_path):
        stats["train_rows"] += 1
        instruction = row["instruction"]
        output = row["output"]
        norm_instruction = normalize_text(instruction)
        if norm_instruction in exact_instructions:
            stats["exact_instruction_hits"] += 1
            if len(stats["examples"]["exact_instruction"]) < max_examples:
                stats["examples"]["exact_instruction"].append(
                    {"instruction": instruction[:300]}
                )
        match = near.find(instruction)
        if match is not None:
            stats["near_instruction_hits"] += 1
            if len(stats["examples"]["near_instruction"]) < max_examples:
                stats["examples"]["near_instruction"].append(
                    {
                        "instruction": instruction[:300],
                        "matched_id": match[0],
                        "similarity": round(match[1], 4),
                    }
                )
        if output and _pair_key(instruction, output) in exact_pairs:
            stats["exact_pair_hits"] += 1
            if len(stats["examples"]["exact_pair"]) < max_examples:
                stats["examples"]["exact_pair"].append(
                    {"instruction": instruction[:300], "output": output[:300]}
                )
        for entity in entities:
            if _whole_phrase_in(entity, instruction) or _whole_phrase_in(
                entity, output
            ):
                stats["entity_hits"] += 1
                if len(stats["examples"]["entity"]) < max_examples:
                    stats["examples"]["entity"].append(
                        {"entity": entity, "instruction": instruction[:300]}
                    )
                break

    stats["total_hits"] = (
        stats["exact_instruction_hits"]
        + stats["near_instruction_hits"]
        + stats["exact_pair_hits"]
        + stats["entity_hits"]
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Training/eval contamination check")
    parser.add_argument("--train", required=True, help="SFT parquet or JSONL")
    parser.add_argument("--eval", default="evaluation/datasets")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument(
        "--entity-seeds", default="distillation/seeds/eval_entities.jsonl"
    )
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--allow-hits",
        action="store_true",
        help="exit 0 even when overlaps are found (inspection only)",
    )
    args = parser.parse_args()

    stats = run_check(
        args.train,
        args.eval,
        threshold=args.threshold,
        entity_seeds=args.entity_seeds,
        max_examples=args.max_examples,
    )
    for key in (
        "exact_instruction_hits",
        "near_instruction_hits",
        "exact_pair_hits",
        "entity_hits",
    ):
        print(f"{key}: {stats[key]:,}")
    print(f"total_hits: {stats['total_hits']:,}")
    for kind, examples in stats["examples"].items():
        for example in examples:
            print(f"  [{kind}] {json.dumps(example, ensure_ascii=False)}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report -> {out}")
    if stats["total_hits"] and not args.allow_hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
