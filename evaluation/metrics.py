"""Rule-based metrics for the frozen evaluation sets.

Scores are returned as ``(score, correct, detail)``:

* ``score`` is a continuous value in ``[0, 1]`` when the check defines one
  (token F1, set F1) and 0/1 for binary checks;
* ``correct`` is the binary accuracy signal used by McNemar;
* ``detail`` stores parsing errors and component scores.

Checks that need a judge (``llm_judge``/``manual``) return ``None`` for both
score and correct.  They are still generated and saved so a human or an
independent judge can score them later.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Any

from distillation.prompts import schema_for_example
from distillation.text_utils import (
    extract_json_list,
    extract_json_value,
    is_refusal,
    normalize_answer,
    normalize_text,
    word_tokens,
)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _aliases(example: dict[str, Any]) -> list[str]:
    aliases = _as_list(example.get("aliases"))
    if aliases:
        return [str(x) for x in aliases if str(x).strip()]
    reference = example.get("reference")
    if isinstance(reference, (list, tuple)):
        return [str(x) for x in reference if str(x).strip()]
    return [str(reference)] if reference is not None and str(reference).strip() else []


def _whole_phrase_in(needle: str, haystack: str) -> bool:
    needle = normalize_text(needle)
    if not needle:
        return False
    return f" {needle} " in f" {normalize_text(haystack)} "


def alias_match(prediction: str, aliases: list[str]) -> bool:
    pred = normalize_answer(prediction)
    if not pred:
        return False
    pred_tokens = pred.split()
    for alias in aliases:
        target = normalize_answer(alias)
        if not target:
            continue
        if pred == target:
            return True
        target_tokens = target.split()
        # Accept a short sentence that contains the answer, but do not accept
        # an entire paragraph that happens to mention it.
        if len(pred_tokens) <= len(target_tokens) + 4:
            if _whole_phrase_in(target, pred):
                return True
            if pred_tokens[: len(target_tokens)] == target_tokens:
                return True
            if pred_tokens[-len(target_tokens) :] == target_tokens:
                return True
    return False


def exact_match(prediction: str, references: list[str]) -> bool:
    pred = normalize_answer(prediction)
    return any(pred == normalize_answer(ref) for ref in references if ref)


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = word_tokens(prediction)
    ref_tokens = word_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def repetition_rate(text: str, n: int = 4) -> float:
    tokens = word_tokens(text)
    if len(tokens) < n:
        return 0.0
    grams = Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    total = sum(grams.values())
    return 1.0 - len(grams) / total if total else 0.0


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")


def _sentence_count(text: str) -> int:
    return len([p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()])


def check_constraint(output: str, constraint: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    name = constraint.get("name")
    value = constraint.get("value")
    detail: dict[str, Any] = {"name": name, "value": value}
    if name == "max_words":
        count = len(word_tokens(output))
        detail["word_count"] = count
        return count <= int(value), detail
    if name == "exact_sentences":
        count = _sentence_count(output)
        detail["sentence_count"] = count
        return count == int(value), detail
    if name == "starts_with":
        ok = output.lower().startswith(str(value).lower())
        detail["start"] = output[:40]
        return ok, detail
    if name == "bullet_count":
        count = sum(1 for line in output.splitlines() if _BULLET_RE.match(line))
        detail["bullet_count"] = count
        return count == int(value), detail
    if name == "includes_word":
        ok = normalize_text(str(value)) in word_tokens(output)
        detail["word"] = value
        return ok, detail
    if name == "lowercase":
        return output == output.lower(), detail
    return False, {**detail, "error": "unknown constraint"}


def score_prediction(
    example: dict[str, Any], prediction: str
) -> tuple[float | None, bool | None, dict[str, Any]]:
    check = str(example.get("check") or "alias_match")
    prediction = prediction or ""
    aliases = _aliases(example)
    reference = example.get("reference")

    if check == "alias_match":
        ok = alias_match(prediction, aliases)
        return (1.0 if ok else 0.0), ok, {"aliases": aliases}
    if check == "exact_match":
        ok = exact_match(prediction, aliases)
        return (1.0 if ok else 0.0), ok, {"aliases": aliases}
    if check == "token_f1":
        refs = aliases or [str(reference)]
        scores = [token_f1(prediction, ref) for ref in refs if ref]
        score = max(scores) if scores else 0.0
        return score, score >= 0.5, {"per_reference": scores}
    if check == "json_schema":
        schema = example.get("schema") or {}
        try:
            value = extract_json_value(prediction)
        except ValueError as exc:
            return 0.0, False, {"error": str(exc)}
        errors = schema_for_example(value, schema)
        return (0.0 if errors else 1.0), not errors, {"errors": errors}
    if check == "set_match":
        refs = [
            normalize_answer(str(x)) for x in _as_list(reference) if str(x).strip()
        ]
        try:
            values = extract_json_list(prediction)
        except ValueError as exc:
            return 0.0, False, {"error": str(exc)}
        pred = {
            normalize_answer(str(x))
            for x in values
            if not isinstance(x, (dict, list)) and str(x).strip()
        }
        ref_set = set(refs)
        if not pred and not ref_set:
            return 1.0, True, {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        overlap = len(pred & ref_set)
        precision = overlap / len(pred) if pred else 0.0
        recall = overlap / len(ref_set) if ref_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return f1, pred == ref_set, {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "predicted": sorted(pred),
            "reference": sorted(ref_set),
        }
    if check == "constraint":
        ok, detail = check_constraint(prediction, example.get("constraint") or {})
        return (1.0 if ok else 0.0), ok, detail
    if check == "regex":
        pattern = str(example.get("pattern") or "")
        ok = bool(re.search(pattern, prediction, flags=example.get("flags", 0)))
        return (1.0 if ok else 0.0), ok, {"pattern": pattern}
    if check == "keyword":
        keywords = [str(x) for x in _as_list(example.get("keywords"))]
        missing = [k for k in keywords if not _whole_phrase_in(k, prediction)]
        ok = bool(keywords) and not missing
        return (1.0 if ok else 0.0), ok, {"missing": missing}
    if check == "nonempty":
        ok = bool(prediction.strip()) and not is_refusal(prediction)
        return (1.0 if ok else 0.0), ok, {}
    if check in {"llm_judge", "manual"}:
        return None, None, {
            "unscored": True,
            "nonempty": bool(prediction.strip()),
            "repetition_rate": round(repetition_rate(prediction), 4),
            "rubric": example.get("rubric", ""),
        }
    raise ValueError(f"unknown evaluation check: {check!r}")


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(max(1, int(n_boot))):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return [round(lo, 4), round(hi, 4)]


def mcnemar(
    baseline_correct: list[bool | None],
    candidate_correct: list[bool | None],
) -> dict[str, Any] | None:
    pairs = [
        (bool(b), bool(c))
        for b, c in zip(baseline_correct, candidate_correct)
        if b is not None and c is not None
    ]
    if not pairs:
        return None
    b = sum(1 for base, cand in pairs if base and not cand)
    c = sum(1 for base, cand in pairs if cand and not base)
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0}
    if n <= 1000:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2 ** n)
        p_value = min(1.0, 2.0 * tail)
        statistic = float(min(b, c))
    else:
        statistic = (abs(b - c) - 1) ** 2 / n
        p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "b": b,
        "c": c,
        "statistic": round(statistic, 4),
        "p_value": round(p_value, 6),
    }


def _summary(rows: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    correct = [bool(r["correct"]) for r in rows if r.get("correct") is not None]
    rep = [
        float(r.get("repetition_rate") or 0.0)
        for r in rows
        if r.get("repetition_rate") is not None
    ]
    return {
        "n": len(rows),
        "scored": len(scores),
        "unscored": len(rows) - len(scores),
        "accuracy": round(sum(correct) / len(correct), 4) if correct else None,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else None,
        "score_ci95": bootstrap_ci(scores, n_boot=n_boot, seed=seed),
        "mean_repetition_rate": round(sum(rep) / len(rep), 4) if rep else None,
    }


def summarize_records(
    records: list[dict[str, Any]], *, n_boot: int = 1000, seed: int = 42
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("category", "")), str(record.get("split", "")))].append(record)

    def summarize_group(rows):
        return _summary(rows, n_boot=n_boot, seed=seed)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (category, split), rows in groups.items():
        by_category[category].extend(rows)
        by_split[split].extend(rows)
        by_pair[f"{category}/{split}"] = rows

    return {
        "overall": summarize_group(records),
        "by_category": {k: summarize_group(v) for k, v in sorted(by_category.items())},
        "by_split": {k: summarize_group(v) for k, v in sorted(by_split.items())},
        "by_category_split": {k: summarize_group(v) for k, v in sorted(by_pair.items())},
    }
