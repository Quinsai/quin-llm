"""Small text helpers shared by data filtering and evaluation.

Everything here is pure Python so the data-preparation stages can run without
importing torch.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable


_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_PREAMBLE_RE = re.compile(
    r"^(sure|of course|certainly|absolutely|here(?:'s| is)|below is)"
    r"[^.\n:]*[.:!]\s*",
    flags=re.IGNORECASE,
)
_ANSWER_PREFIX_RE = re.compile(r"^(answer|response)\s*:\s*", flags=re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n```\s*$", flags=re.DOTALL)
_REFUSAL_RE = re.compile(
    r"\b(i'?m sorry|i am sorry|i can'?t|i cannot|as an ai|i am unable|"
    r"unable to (?:answer|help)|i won'?t)\b",
    flags=re.IGNORECASE,
)


def normalize_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").strip())


def normalize_text(text: str) -> str:
    """Lowercase, NFKC-normalize, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return normalize_whitespace(text)


def normalize_answer(text: str) -> str:
    """Normalize an answer for exact/alias comparison."""
    tokens = [tok for tok in normalize_text(text).split() if tok not in _ARTICLES]
    return " ".join(tokens)


def english_ratio(text: str) -> float:
    """Fraction of visible characters that are ASCII.

    The student is English-only; this cheaply rejects mostly non-ASCII output
    (for example Chinese or emoji-heavy answers).
    """
    visible = [ch for ch in text or "" if not ch.isspace()]
    if not visible:
        return 0.0
    return sum(ch.isascii() for ch in visible) / len(visible)


def word_tokens(text: str) -> list[str]:
    return normalize_text(text).split()


def shingles(text: str, n: int = 5) -> frozenset[tuple[str, ...]]:
    """Return a set of word n-grams used for near-duplicate detection."""
    tokens = word_tokens(text)
    if not tokens:
        return frozenset({("__empty__",)})
    if len(tokens) < n:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


class NearDuplicateIndex:
    """Inverted-index near-duplicate detector.

    A naive all-pairs comparison over tens of thousands of samples is too
    slow.  Indexing each shingle to the records that contain it limits the
    Jaccard comparison to plausible candidates.
    """

    def __init__(self, n: int = 5, threshold: float = 0.85) -> None:
        self.n = int(n)
        self.threshold = float(threshold)
        self._index: dict[tuple[str, ...], set[str]] = defaultdict(set)
        self._shingles: dict[str, frozenset] = {}

    def add(self, text: str, key: str) -> None:
        sh = shingles(text, self.n)
        self._shingles[key] = sh
        for shingle in sh:
            self._index[shingle].add(key)

    def find(self, text: str) -> tuple[str, float] | None:
        sh = shingles(text, self.n)
        candidates: set[str] = set()
        for shingle in sh:
            candidates.update(self._index.get(shingle, ()))

        best_key: str | None = None
        best_score = 0.0
        for key in candidates:
            score = jaccard(sh, self._shingles[key])
            if score > best_score:
                best_key, best_score = key, score
        if best_key is not None and best_score >= self.threshold:
            return best_key, best_score
        return None

    def __len__(self) -> int:
        return len(self._shingles)


def sentence_count(text: str) -> int:
    parts = [p for p in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if p.strip()]
    return len(parts)


def is_refusal(text: str) -> bool:
    return bool(_REFUSAL_RE.search(text or ""))


def strip_preamble(text: str) -> str:
    """Remove common chat-assistant prefixes such as ``Sure, ...``."""
    out = (text or "").strip()
    for _ in range(3):
        new = _PREAMBLE_RE.sub("", out, count=1)
        new = _ANSWER_PREFIX_RE.sub("", new, count=1)
        if new == out:
            break
        out = new.strip()
    return out


def strip_markdown_fence(text: str) -> str:
    match = _FENCE_RE.match(text or "")
    return match.group(1).strip() if match else (text or "").strip()


def extract_json_value(text: str) -> Any:
    """Extract the first JSON object/array from a model response.

    Teacher responses sometimes wrap JSON in prose or a markdown fence.  We
    try the whole string first, then the substring between the first opening
    and last matching closing bracket.
    """
    cleaned = strip_markdown_fence((text or "").strip())
    candidates = [cleaned]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        end = cleaned.rfind(closer)
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"no valid JSON found in response: {cleaned[:200]!r}")


def extract_json_object(text: str) -> dict[str, Any]:
    value = extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object, got {type(value).__name__}")
    return value


def extract_json_list(text: str) -> list[Any]:
    value = extract_json_value(text)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON array, got {type(value).__name__}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def text_field(record: dict[str, Any], *names: str) -> str:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
