"""Filter raw teacher outputs into training-ready SFT rows.

Design goals:

* deterministic: records are processed in ``(category, task_id)`` order, so
  the same raw file and config produce the same filtered file;
* conservative: a row is kept only when it parses, matches its category's
  contract, is long enough to be useful, short enough to survive ``prepare_sft``,
  and does not duplicate or contaminate evaluation data;
* auditable: every rejected row records all reasons, and a stratified sample
  of kept (plus a small sample of rejected) rows is written to a review CSV.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from config import DistillFilterConfig
from pipeline.tokenizer.bpe_tokenizer import BPETokenizer

from .prompts import build_user_prompt, schema_for_example
from .text_utils import (
    NearDuplicateIndex,
    english_ratio,
    extract_json_list,
    extract_json_object,
    extract_json_value,
    hash_text,
    is_refusal,
    iter_jsonl,
    normalize_answer,
    normalize_text,
    sentence_count,
    strip_preamble,
    word_tokens,
)


PROMPT_OVERHEAD_TOKENS = 12
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_STEP_RE = re.compile(r"\b(step\s*1|step-by-step|chain of thought)\b", re.IGNORECASE)
_DONT_KNOW_RE = re.compile(r"^\s*i\s+(?:do not|don't)\s+know\b", re.IGNORECASE)
_FAILED_CATEGORIES = {"fact_qa", "format_json", "extraction", "instruction", "reasoning"}
# Only entities from these eval splits were deliberately held out.  Seen-entity
# splits mention entities that are supposed to appear in training data.
# External-floor items are guarded by exact/near matching instead: their
# subjects include generic words ("gold", "1984", "world") that would trigger
# false-positive entity hits.
_HELD_OUT_ENTITY_SPLITS = {"unseen_entity"}


# ----------------------------------------------------------------------
# text checks
# ----------------------------------------------------------------------
def _phrase_in(phrase: str, text: str) -> bool:
    needle = normalize_text(phrase)
    if not needle:
        return False
    return f" {needle} " in f" {normalize_text(text)} "


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _contains_any(text: str, values: list[Any]) -> bool:
    return any(_phrase_in(str(value), text) for value in values if str(value).strip())


def _clean_output(text: str) -> str:
    out = strip_preamble((text or "").replace("\r\n", "\n").replace("\r", "\n"))
    out = out.strip()
    out = re.sub(r"\n{3,}", "\n\n", out)
    if "\n" not in out and len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        out = out[1:-1].strip()
    return out


def _is_degenerate(text: str) -> bool:
    tokens = word_tokens(text)
    if len(tokens) >= 8 and len(set(tokens)) / len(tokens) < 0.35:
        return True
    if len(tokens) >= 12:
        grams = Counter(
            tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)
        )
        if grams and max(grams.values()) >= 3:
            return True
    if len(text) > 30 and len(set(text)) <= 3:
        return True
    return False


def _bullet_count(text: str) -> int:
    return sum(1 for line in text.splitlines() if _BULLET_RE.match(line))


_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_COUNT_UNIT = r"(?:words?|sentences?|bullet\s+points?|bullets?)"
_QUALIFIED_COUNT_RE = re.compile(
    rf"\b(exactly|at\s+most|at\s+least|no\s+more\s+than|no\s+fewer\s+than|"
    rf"maximum\s+of|minimum\s+of|up\s+to)\s+"
    rf"(\d+|{'|'.join(_NUMBER_WORDS)})(?:\s+(?:short|simple|brief|complete|"
    rf"full|total|separate|distinct))?\s+({_COUNT_UNIT})\b",
    re.IGNORECASE,
)
_OR_COUNT_RE = re.compile(
    rf"\b(\d+|{'|'.join(_NUMBER_WORDS)})(?:\s+(?:short|simple|brief|complete|"
    rf"full|total|separate|distinct))?\s+({_COUNT_UNIT})\s+or\s+(fewer|less|more)\b",
    re.IGNORECASE,
)


def _parse_count(value: str) -> int | None:
    value = value.lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def _is_prompt_echo(instruction: str, task: dict[str, Any]) -> bool:
    """True when the teacher copied the meta-prompt into ``instruction``."""
    try:
        prompt = build_user_prompt(task)
    except (KeyError, TypeError, ValueError):
        return False
    norm_instruction = normalize_text(instruction)
    norm_prompt = normalize_text(prompt)
    if not norm_instruction or not norm_prompt:
        return False
    if norm_instruction == norm_prompt:
        return True
    if len(norm_instruction) >= 40 and norm_instruction in norm_prompt:
        return True
    if len(norm_prompt) >= 40 and norm_prompt in norm_instruction:
        return True
    return False


def _check_declared_constraints(
    instruction: str, output: str, reasons: list[str], details: dict[str, Any]
) -> None:
    """Check numeric constraints the teacher wrote into the instruction itself.

    Seed constraints are checked separately; this catches cases such as an
    instruction saying "exactly 15 words" when the answer has 19.
    """
    checks: list[tuple[int, int, str, str]] = []
    for match in _QUALIFIED_COUNT_RE.finditer(instruction):
        qualifier, number_text = match.group(1), match.group(2)
        unit = match.group(3).lower()
        number = _parse_count(number_text)
        if number is None:
            continue
        checks.append((number, _count_for_unit(unit, output), qualifier, unit))
    for match in _OR_COUNT_RE.finditer(instruction):
        number_text = match.group(1)
        unit = match.group(2).lower()
        direction = match.group(3).lower()
        number = _parse_count(number_text)
        if number is None:
            continue
        qualifier = "at most" if direction in {"fewer", "less"} else "at least"
        checks.append((number, _count_for_unit(unit, output), qualifier, unit))

    for expected, actual, qualifier, unit in checks:
        normalized = " ".join(qualifier.lower().split())
        failed = False
        if normalized == "exactly":
            failed = actual != expected
        elif normalized in {"at most", "no more than", "maximum of", "up to"}:
            failed = actual > expected
        elif normalized in {"at least", "no fewer than", "minimum of"}:
            failed = actual < expected
        if failed:
            reasons.append(f"constraint_declared_{_unit_name(unit)}")
            details[f"declared_{_unit_name(unit)}"] = {
                "expected": expected,
                "actual": actual,
                "qualifier": normalized,
            }


def _count_for_unit(unit: str, output: str) -> int:
    if unit.startswith("word"):
        return len(word_tokens(output))
    if unit.startswith("sentence"):
        return sentence_count(output)
    return _bullet_count(output)


def _unit_name(unit: str) -> str:
    if unit.startswith("word"):
        return "words"
    if unit.startswith("sentence"):
        return "sentences"
    return "bullets"


# ----------------------------------------------------------------------
# tokenizer
# ----------------------------------------------------------------------
class TokenCounter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.tokenizer: BPETokenizer | None = None
        if Path(path).exists():
            self.tokenizer = BPETokenizer.load(path)
        else:
            print(
                f"warning: tokenizer not found at {path}; length checks use words"
            )

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_bos=False, add_eos=False))
        return len(word_tokens(text))


# ----------------------------------------------------------------------
# evaluation contamination
# ----------------------------------------------------------------------
class ContaminationIndex:
    def __init__(
        self,
        eval_dir: str,
        entity_seeds: str,
        threshold: float,
    ) -> None:
        self.enabled = False
        self.files: list[str] = []
        self.samples = 0
        self.exact: set[str] = set()
        self.pairs: set[str] = set()
        self.entities: set[str] = set()
        self.near = NearDuplicateIndex(threshold=threshold)
        self._near_meta: dict[str, str] = {}
        self.hits = Counter()

        for path in self._eval_files(eval_dir):
            self.files.append(str(path))
            for record in iter_jsonl(str(path)):
                instruction = _text(record, "instruction", "question", "prompt")
                reference = _text(record, "reference", "output", "answer")
                if not instruction:
                    continue
                self.samples += 1
                sample_id = str(record.get("id") or f"{path.name}:{self.samples}")
                self.exact.add(normalize_text(instruction))
                if reference:
                    self.pairs.add(self._pair_key(instruction, reference))
                self.near.add(instruction, sample_id)
                self._near_meta[sample_id] = str(path)
                if str(record.get("split", "")) in _HELD_OUT_ENTITY_SPLITS:
                    metadata = record.get("metadata") or {}
                    entity = metadata.get("entity") or record.get("entity")
                    if entity:
                        self.entities.add(str(entity))

        seed_path = Path(entity_seeds)
        if seed_path.exists():
            for item in iter_jsonl(str(seed_path)):
                entity = str(item.get("entity", "")).strip()
                if entity:
                    self.entities.add(entity)

        self.enabled = bool(self.samples or self.entities)

    @classmethod
    def disabled(cls) -> "ContaminationIndex":
        obj = cls.__new__(cls)
        obj.enabled = False
        obj.files = []
        obj.samples = 0
        obj.exact = set()
        obj.pairs = set()
        obj.entities = set()
        obj.near = NearDuplicateIndex()
        obj._near_meta = {}
        obj.hits = Counter()
        return obj

    @staticmethod
    def _eval_files(eval_dir: str) -> list[Path]:
        root = Path(eval_dir)
        if not root.exists():
            return []
        if root.is_file():
            return [root]
        return sorted(root.rglob("*.jsonl"))

    @staticmethod
    def _pair_key(instruction: str, output: str) -> str:
        return normalize_text(instruction) + "\n" + normalize_answer(output)

    def check(self, instruction: str, output: str) -> list[tuple[str, dict[str, Any]]]:
        if not self.enabled:
            return []
        found: list[tuple[str, dict[str, Any]]] = []
        if normalize_text(instruction) in self.exact:
            found.append(("eval_exact_instruction", {}))
        if self._pair_key(instruction, output) in self.pairs:
            found.append(("eval_exact_pair", {}))
        near = self.near.find(instruction)
        if near is not None:
            key, score = near
            found.append(
                (
                    "eval_near_duplicate",
                    {"matched_id": key, "similarity": round(score, 4)},
                )
            )
        for entity in self.entities:
            if _phrase_in(entity, instruction) or _phrase_in(entity, output):
                found.append(("eval_entity_mentioned", {"entity": entity}))
                break
        for reason, _ in found:
            self.hits[reason] += 1
        return found


def _text(record: dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


# ----------------------------------------------------------------------
# category contracts
# ----------------------------------------------------------------------
def _check_fact(
    instruction: str, output: str, seed: dict[str, Any], reasons: list[str], details: dict
) -> None:
    entity = str(seed.get("entity", "")).strip()
    expected = [str(x) for x in _as_list(seed.get("expected")) if str(x).strip()]
    if entity and not _phrase_in(entity, instruction):
        reasons.append("entity_missing_from_instruction")
    if expected:
        if _contains_any(instruction, expected):
            reasons.append("answer_in_instruction")
        if not _contains_any(output, expected):
            reasons.append("fact_mismatch")
            details["expected"] = expected
    if sentence_count(output) > 2:
        reasons.append("too_many_sentences")
    if output.endswith("?"):
        reasons.append("answer_is_question")


def _check_json(
    instruction: str, output: str, seed: dict[str, Any], reasons: list[str], details: dict
) -> str:
    if "json" not in instruction.lower():
        reasons.append("json_not_requested")
    try:
        value = extract_json_value(output)
    except ValueError:
        reasons.append("invalid_json_output")
        return output
    schema = seed.get("schema")
    if isinstance(schema, dict):
        errors = schema_for_example(value, schema)
        if errors:
            reasons.append("schema_mismatch")
            details["schema_errors"] = errors
    return json.dumps(value, ensure_ascii=False)


def _check_extraction(
    instruction: str, output: str, reasons: list[str], details: dict
) -> str:
    try:
        values = extract_json_list(output)
    except ValueError:
        reasons.append("invalid_json_output")
        return output
    normalized: list[str] = []
    for value in values:
        if isinstance(value, (dict, list)) or value is None:
            reasons.append("bad_extraction_value")
            continue
        normalized.append(str(value))
    if len(normalized) != 1:
        reasons.append("extraction_count")
        details["extraction_count"] = len(normalized)
    if normalized and not _contains_any(instruction, normalized):
        reasons.append("extraction_not_in_instruction")
    return json.dumps(normalized, ensure_ascii=False)


def _check_instruction_constraint(
    output: str, seed: dict[str, Any], reasons: list[str], details: dict
) -> None:
    constraint = seed.get("constraint") or {}
    name = constraint.get("name")
    value = constraint.get("value")
    if not name:
        return
    if name == "max_words":
        count = len(word_tokens(output))
        if count > int(value):
            reasons.append("constraint_max_words")
            details["word_count"] = count
    elif name == "exact_sentences":
        count = sentence_count(output)
        if count != int(value):
            reasons.append("constraint_exact_sentences")
            details["sentence_count"] = count
    elif name == "starts_with":
        if not output.lower().startswith(str(value).lower()):
            reasons.append("constraint_starts_with")
    elif name == "bullet_count":
        count = _bullet_count(output)
        if count != int(value):
            reasons.append("constraint_bullet_count")
            details["bullet_count"] = count
    elif name == "includes_word":
        if normalize_text(str(value)) not in word_tokens(output):
            reasons.append("constraint_includes_word")
    elif name == "lowercase":
        if output != output.lower():
            reasons.append("constraint_lowercase")


def _check_reasoning(output: str, reasons: list[str]) -> None:
    if sentence_count(output) > 4:
        reasons.append("too_many_sentences")
    if _STEP_RE.search(output):
        reasons.append("chain_of_thought")


# ----------------------------------------------------------------------
# filtering
# ----------------------------------------------------------------------
def _load_raw(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping malformed raw line {path}:{line_no}")
    records.sort(key=lambda r: (str(r.get("category", "")), str(r.get("task_id", ""))))
    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_record(raw: dict[str, Any], instruction: str, output: str) -> dict[str, Any]:
    seed = raw.get("seed") or {}
    return {
        "id": str(raw.get("task_id", "")),
        "task_id": str(raw.get("task_id", "")),
        "instruction": instruction,
        "input": "",
        "output": output,
        "category": str(raw.get("category", "")),
        "difficulty": str(raw.get("difficulty", "")),
        "seed": seed,
        "entity": str(seed.get("entity", "")),
        "topic": str(seed.get("topic", "")),
        "expected": _as_list(seed.get("expected")),
        "source": "teacher",
        "seed_source": str(raw.get("seed_source", "")),
        "model": str(raw.get("model", "")),
        "prompt_version": str(raw.get("prompt_version", "")),
        "request_id": str(raw.get("request_id", "")),
    }


def _filter_one(
    raw: dict[str, Any],
    counter: TokenCounter,
    cfg: DistillFilterConfig,
    contamination: ContaminationIndex,
    seen_instructions: set[str],
    seen_pairs: set[str],
    near: NearDuplicateIndex,
    caps: Counter,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any]]:
    reasons: list[str] = []
    details: dict[str, Any] = {}
    content = str(raw.get("content") or "")

    if raw.get("status") != "ok":
        reasons.append("api_error")
    finish = raw.get("finish_reason")
    if finish not in (None, "stop"):
        reasons.append(f"finish_reason_{finish}")

    instruction = ""
    output = ""
    try:
        obj = extract_json_object(content)
        instruction = str(obj.get("instruction") or "").strip()
        output = obj.get("output")
        category = str(raw.get("category", ""))
        if not isinstance(output, str):
            if category in {"format_json", "extraction"}:
                output = json.dumps(output, ensure_ascii=False)
            else:
                reasons.append("output_not_string")
                output = ""
        output = _clean_output(str(output))
    except (ValueError, TypeError):
        reasons.append("invalid_json")

    if not instruction:
        reasons.append("empty_instruction")
    if not output:
        reasons.append("empty_output")

    category = str(raw.get("category", ""))
    seed = raw.get("seed") or {}
    if category not in _FAILED_CATEGORIES:
        reasons.append("unknown_category")
    elif instruction and _is_prompt_echo(
        instruction,
        {
            "category": category,
            "seed": seed,
            "difficulty": raw.get("difficulty", "easy"),
        },
    ):
        reasons.append("prompt_echo")

    if instruction:
        if len(instruction) < int(cfg.min_instruction_chars):
            reasons.append("instruction_too_short")
        if english_ratio(instruction) < float(cfg.min_english_ratio):
            reasons.append("instruction_not_english")
    if output:
        if english_ratio(output) < float(cfg.min_english_ratio):
            reasons.append("output_not_english")
        if is_refusal(output) or _DONT_KNOW_RE.search(output):
            reasons.append("refusal")
        if output == instruction or normalize_text(output) == normalize_text(instruction):
            reasons.append("output_repeats_instruction")
        if _is_degenerate(output):
            reasons.append("repetitive")
        if len(output) > int(cfg.max_output_chars):
            reasons.append("output_too_long")

    # Category-specific cleaning and rule checks.
    if output and category == "format_json":
        output = _check_json(instruction, output, seed, reasons, details)
    elif output and category == "extraction":
        output = _check_extraction(instruction, output, reasons, details)
    elif output and category == "fact_qa":
        _check_fact(instruction, output, seed, reasons, details)
    elif output and category == "instruction":
        _check_instruction_constraint(output, seed, reasons, details)
        _check_declared_constraints(instruction, output, reasons, details)
    elif output and category == "reasoning":
        _check_reasoning(output, reasons)

    instruction_tokens = counter.count(instruction) if instruction else 0
    output_tokens = counter.count(output) if output else 0
    total_est = instruction_tokens + output_tokens + PROMPT_OVERHEAD_TOKENS
    if instruction_tokens > int(cfg.max_instruction_tokens):
        reasons.append("instruction_too_long")
    if output_tokens > int(cfg.max_output_tokens):
        reasons.append("output_tokens_too_long")
    if total_est > int(cfg.block_size):
        reasons.append("overlong")

    # Deduplication and distribution caps only apply to otherwise valid rows.
    if not reasons and instruction:
        norm_instruction = normalize_text(instruction)
        pair_key = hash_text(norm_instruction + "\n" + normalize_answer(output))
        if norm_instruction in seen_instructions:
            reasons.append("duplicate_instruction")
        elif pair_key in seen_pairs:
            reasons.append("duplicate_pair")
        else:
            near_match = near.find(instruction + "\n" + output)
            if near_match is not None:
                key, score = near_match
                reasons.append("near_duplicate")
                details["near_duplicate_of"] = key
                details["similarity"] = round(score, 4)
            cap_key = str(seed.get("entity") or seed.get("topic") or "").strip()
            if cap_key and int(cfg.max_per_entity) > 0:
                if caps[normalize_text(cap_key)] >= int(cfg.max_per_entity):
                    reasons.append("entity_cap")
                    details["cap_key"] = cap_key

    if not reasons:
        for reason, info in contamination.check(instruction, output):
            reasons.append(reason)
            details.update(info)

    base = _base_record(raw, instruction, output)
    base["tokens"] = {
        "instruction": instruction_tokens,
        "output": output_tokens,
        "total_est": total_est,
    }

    if reasons:
        rejected = dict(base)
        rejected["reasons"] = reasons
        rejected["details"] = details
        if not instruction and content:
            rejected["raw_content"] = content[:800]
        return None, rejected, {"reasons": reasons}

    norm_instruction = normalize_text(instruction)
    pair_key = hash_text(norm_instruction + "\n" + normalize_answer(output))
    seen_instructions.add(norm_instruction)
    seen_pairs.add(pair_key)
    near.add(instruction + "\n" + output, base["id"])
    cap_key = str(seed.get("entity") or seed.get("topic") or "").strip()
    if cap_key:
        caps[normalize_text(cap_key)] += 1
    return base, {}, {}


def filter_records(
    raw_path: str | Path,
    out_path: str | Path,
    cfg: DistillFilterConfig,
    *,
    rejected_path: str | Path | None = None,
    stats_path: str | Path | None = None,
    review_path: str | Path | None = None,
    contamination: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    raw_file = Path(raw_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rejected_file = (
        Path(rejected_path)
        if rejected_path
        else out_file.with_name(out_file.stem + ".rejected.jsonl")
    )
    stats_file = (
        Path(stats_path)
        if stats_path
        else out_file.with_name(out_file.stem + ".stats.json")
    )
    review_file = (
        Path(review_path)
        if review_path
        else out_file.with_name(out_file.stem + ".review.csv")
    )

    records = _load_raw(raw_file)
    if limit is not None:
        records = records[: max(0, int(limit))]
    counter = TokenCounter(cfg.tokenizer_path)
    if contamination:
        contamination_index = ContaminationIndex(
            cfg.eval_dir, cfg.eval_entity_seeds, cfg.near_dup_threshold
        )
        if not contamination_index.enabled:
            print(
                "warning: no evaluation data found at "
                f"{cfg.eval_dir!r}; contamination checks are disabled"
            )
    else:
        contamination_index = ContaminationIndex.disabled()

    near = NearDuplicateIndex(threshold=cfg.near_dup_threshold)
    seen_instructions: set[str] = set()
    seen_pairs: set[str] = set()
    caps: Counter = Counter()
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter = Counter()
    category_kept: Counter = Counter()
    category_rejected: Counter = Counter()
    difficulty_kept: Counter = Counter()

    for raw in records:
        base, reject, meta = _filter_one(
            raw,
            counter,
            cfg,
            contamination_index,
            seen_instructions,
            seen_pairs,
            near,
            caps,
        )
        if base is not None:
            kept.append(base)
            category_kept[base["category"]] += 1
            difficulty_kept[base["difficulty"]] += 1
        else:
            rejected.append(reject)
            category_rejected[reject["category"]] += 1
            for reason in meta["reasons"]:
                reason_counts[reason] += 1

    with open(out_file, "w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(rejected_file, "w", encoding="utf-8") as f:
        for record in rejected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def lengths(records_: list[dict[str, Any]], field: str) -> dict[str, float]:
        values = [int((r.get("tokens") or {}).get(field, 0)) for r in records_]
        if not values:
            return {"avg": 0.0, "max": 0}
        return {"avg": round(sum(values) / len(values), 2), "max": max(values)}

    stats = {
        "raw_path": str(raw_file),
        "raw_sha256": _file_sha256(raw_file),
        "limit": limit,
        "raw_records": len(records),
        "kept": len(kept),
        "rejected": len(rejected),
        "kept_fraction": round(len(kept) / len(records), 4) if records else 0.0,
        "reasons": dict(reason_counts.most_common()),
        "categories_raw": dict(Counter(str(r.get("category", "")) for r in records)),
        "categories_kept": dict(category_kept),
        "categories_rejected": dict(category_rejected),
        "difficulty_kept": dict(difficulty_kept),
        "lengths_kept": {
            "instruction": lengths(kept, "instruction"),
            "output": lengths(kept, "output"),
            "total_est": lengths(kept, "total_est"),
        },
        "overlong": reason_counts.get("overlong", 0),
        "duplicates": {
            "exact_instruction": reason_counts.get("duplicate_instruction", 0),
            "exact_pair": reason_counts.get("duplicate_pair", 0),
            "near": reason_counts.get("near_duplicate", 0),
        },
        "entity_cap_rejections": reason_counts.get("entity_cap", 0),
        "contamination": {
            "enabled": bool(contamination_index.enabled),
            "eval_files": list(contamination_index.files),
            "eval_samples": int(contamination_index.samples),
            "eval_entities": len(contamination_index.entities),
            "hits": dict(contamination_index.hits),
        },
        "tokenizer": {
            "path": cfg.tokenizer_path,
            "available": counter.tokenizer is not None,
        },
        "config": cfg.to_dict(),
        "review": {},
    }

    review_counts = _write_review(
        kept, rejected, review_file, cfg
    )
    stats["review"] = {
        "path": str(review_file),
        "kept_sampled": review_counts["kept"],
        "rejected_sampled": review_counts["rejected"],
    }
    stats_file.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"filtered: kept {len(kept):,} / {len(records):,} -> {out_file}")
    print(f"  rejected: {len(rejected):,} -> {rejected_file}")
    if reason_counts:
        top = ", ".join(f"{k}={v:,}" for k, v in reason_counts.most_common(8))
        print(f"  top reasons: {top}")
    print(f"  stats: {stats_file}")
    print(
        f"  review: {review_counts['kept']:,} kept + "
        f"{review_counts['rejected']:,} rejected -> {review_file}"
    )
    return stats


def _stratified_sample(
    records: list[dict[str, Any]],
    fraction: float,
    minimum: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not records:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("category", ""))].append(record)
    chosen: dict[str, dict[str, Any]] = {}
    for category in sorted(groups):
        rows = groups[category]
        n = max(1, int(round(len(rows) * fraction)))
        for record in rng.sample(rows, min(n, len(rows))):
            chosen[record["id"]] = record
    if len(chosen) < minimum:
        pool = [r for r in records if r["id"] not in chosen]
        extra = min(minimum - len(chosen), len(pool))
        for record in rng.sample(pool, extra):
            chosen[record["id"]] = record
    return sorted(chosen.values(), key=lambda r: (str(r.get("category", "")), r["id"]))


def write_review_csv(
    kept: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    path: str | Path,
    *,
    fraction: float,
    min_samples: int,
    rejected_fraction: float,
    min_rejected: int,
    seed: int,
) -> dict[str, int]:
    path = Path(path)
    rng = random.Random(seed)
    kept_sample = _stratified_sample(
        kept, float(fraction), int(min_samples), rng
    )
    rejected_sample = _stratified_sample(
        rejected,
        float(rejected_fraction),
        int(min_rejected),
        rng,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "set",
        "sample_id",
        "category",
        "difficulty",
        "instruction",
        "output",
        "expected",
        "auto_reasons",
        "model_verdict",
        "model_notes",
        "human_verdict",
        "human_notes",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for label, rows in (("kept", kept_sample), ("rejected", rejected_sample)):
            for record in rows:
                writer.writerow(
                    {
                        "set": label,
                        "sample_id": record.get("id", ""),
                        "category": record.get("category", ""),
                        "difficulty": record.get("difficulty", ""),
                        "instruction": record.get("instruction", ""),
                        "output": record.get("output", ""),
                        "expected": json.dumps(
                            record.get("expected") or [], ensure_ascii=False
                        ),
                        "auto_reasons": ";".join(record.get("reasons") or []),
                        "model_verdict": "",
                        "model_notes": "",
                        "human_verdict": "",
                        "human_notes": "",
                    }
                )
    return {"kept": len(kept_sample), "rejected": len(rejected_sample)}


def _write_review(
    kept: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    path: Path,
    cfg: DistillFilterConfig,
) -> dict[str, int]:
    return write_review_csv(
        kept,
        rejected,
        path,
        fraction=cfg.review_fraction,
        min_samples=cfg.min_review_samples,
        rejected_fraction=cfg.rejected_review_fraction,
        min_rejected=cfg.min_rejected_review,
        seed=cfg.seed,
    )
