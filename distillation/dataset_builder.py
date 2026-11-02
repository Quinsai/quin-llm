"""Mix filtered teacher data with Alpaca and write an SFT parquet.

The mix is controlled by **answer-token mass**, not row count.  Alpaca answers
are much shorter than teacher answers, so a 50/50 row split would not be a
50/50 token split.  We keep all teacher rows and take enough Alpaca rows to
reach ``teacher_ratio`` of answer tokens; if Alpaca is the limiting side, we
keep all Alpaca rows and report the lower achieved ratio.

The output columns ``instruction`` / ``input`` / ``output`` are exactly what
``pipeline.sources.iter_sft_examples`` reads.  Extra metadata columns are
ignored by the existing SFT pipeline.
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from config import DatasetBuildConfig
from pipeline.tokenizer.bpe_tokenizer import BPETokenizer
from utils.tqdm_utils import PROGRESS_BAR_FORMAT, progress_kwargs

from .text_utils import iter_jsonl


def _expand_paths(pattern: str | list[str]) -> list[Path]:
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    paths: list[Path] = []
    for item in patterns:
        matches = sorted(glob.glob(str(item)))
        paths.extend(Path(m) for m in (matches or [str(item)]))
    return paths


def _file_info(paths: Iterable[Path]) -> list[dict[str, Any]]:
    info = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            info.append(
                {
                    "path": str(path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            )
        else:
            info.append({"path": str(path), "missing": True})
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


class TokenCounter:
    def __init__(self, path: str) -> None:
        self.path = path
        self.tokenizer = None
        if Path(path).exists():
            self.tokenizer = BPETokenizer.load(path)
        else:
            print(f"warning: tokenizer not found at {path}; counting words")
        self.cache: dict[str, int] = {}

    def count_answer(self, text: str) -> int:
        cached = self.cache.get(text)
        if cached is not None:
            return cached
        if self.tokenizer is not None:
            count = len(self.tokenizer.encode(text, add_bos=False, add_eos=True))
        else:
            count = len(text.split()) + 1
        if len(self.cache) < 200_000:
            self.cache[text] = count
        return count


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _load_alpaca_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Alpaca source not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            records: Iterable[dict[str, Any]] = iter_jsonl(str(path))
        elif suffix in {".parquet", ".pq"}:
            from datasets import load_dataset

            records = load_dataset(
                "parquet", data_files=str(path), split="train"
            )
        else:
            raise ValueError(f"unsupported Alpaca file type: {path}")
        for record in records:
            rows.append(
                {
                    "instruction": _as_text(record.get("instruction")),
                    "input": _as_text(record.get("input")),
                    "output": _as_text(record.get("output")),
                }
            )
    return rows


def _prepare_alpaca(
    rows: list[dict[str, str]], counter: TokenCounter
) -> list[dict[str, Any]]:
    prepared = []
    for index, row in enumerate(
        tqdm(
            rows,
            desc="alpaca-tokens",
            unit="row",
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
            **progress_kwargs(),
        )
    ):
        if not row["instruction"] or not row["output"]:
            continue
        user = row["instruction"] + ("\n" + row["input"] if row["input"] else "")
        prepared.append(
            {
                "instruction": user,
                "input": "",
                "output": row["output"],
                "source": "alpaca",
                "category": "",
                "difficulty": "",
                "task_id": f"alpaca-{index:06d}",
                "answer_tokens": counter.count_answer(row["output"]),
            }
        )
    return prepared


def _prepare_teacher(
    rows: list[dict[str, Any]], counter: TokenCounter
) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        instruction = str(row.get("instruction", "")).strip()
        answer = str(row.get("output", "")).strip()
        if not instruction or not answer:
            continue
        prepared.append(
            {
                "instruction": instruction,
                "input": str(row.get("input", "")).strip(),
                "output": answer,
                "source": "teacher",
                "category": str(row.get("category", "")),
                "difficulty": str(row.get("difficulty", "")),
                "task_id": str(row.get("task_id") or row.get("id") or ""),
                "answer_tokens": counter.count_answer(answer),
            }
        )
    return prepared


def _take_until_tokens(
    rows: list[dict[str, Any]], target_tokens: int
) -> tuple[list[dict[str, Any]], int]:
    if target_tokens <= 0 or not rows:
        return [], 0
    selected: list[dict[str, Any]] = []
    total = 0
    for row in rows:
        selected.append(row)
        total += int(row["answer_tokens"])
        if total >= target_tokens:
            break
    return selected, total


def _cap_pool(
    alpaca: list[dict[str, Any]],
    teacher: list[dict[str, Any]],
    target: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not target or len(alpaca) + len(teacher) <= target:
        return alpaca, teacher
    total = len(alpaca) + len(teacher)
    n_alpaca = max(0, round(len(alpaca) * target / total))
    n_alpaca = min(n_alpaca, len(alpaca))
    n_teacher = min(target - n_alpaca, len(teacher))
    # If one side was rounded away, fill from the other.
    if n_teacher <= 0 and teacher:
        n_teacher = 1
        n_alpaca = min(len(alpaca), max(0, target - 1))
    if n_alpaca <= 0 and alpaca:
        n_alpaca = 1
        n_teacher = min(len(teacher), max(0, target - 1))
    return alpaca[:n_alpaca], teacher[:n_teacher]


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def run_build_dataset(cfg: DatasetBuildConfig) -> dict[str, Any]:
    if not cfg.filtered_path:
        raise ValueError("filtered_path is required")
    filtered_path = Path(cfg.filtered_path)
    if not filtered_path.exists():
        raise FileNotFoundError(f"filtered data not found: {filtered_path}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = TokenCounter(cfg.tokenizer_path)

    teacher_raw = list(iter_jsonl(str(filtered_path)))
    teacher = _prepare_teacher(teacher_raw, counter)
    alpaca_paths = _expand_paths(cfg.alpaca_paths)
    alpaca = _prepare_alpaca(_load_alpaca_rows(alpaca_paths), counter)

    if not teacher and float(cfg.teacher_ratio) > 0:
        raise ValueError("no filtered teacher rows; nothing to mix")
    if not alpaca and float(cfg.teacher_ratio) < 1:
        raise ValueError("no Alpaca rows; cannot build a mixed dataset")

    rng = random.Random(cfg.seed)
    rng.shuffle(alpaca)
    rng.shuffle(teacher)

    teacher_tokens = sum(int(r["answer_tokens"]) for r in teacher)
    alpaca_tokens_total = sum(int(r["answer_tokens"]) for r in alpaca)
    ratio = max(0.0, min(1.0, float(cfg.teacher_ratio)))

    if ratio <= 0:
        alpaca_sel, alpaca_tokens = _take_until_tokens(alpaca, alpaca_tokens_total)
        teacher_sel, teacher_tokens = [], 0
    elif ratio >= 1:
        alpaca_sel, alpaca_tokens = [], 0
        teacher_sel = teacher
    else:
        desired_alpaca_tokens = int(teacher_tokens * (1.0 - ratio) / ratio)
        if desired_alpaca_tokens >= alpaca_tokens_total:
            alpaca_sel, alpaca_tokens = alpaca, alpaca_tokens_total
            print(
                "warning: not enough Alpaca tokens to reach the requested "
                f"teacher ratio {ratio:.2f}; using all Alpaca rows"
            )
        else:
            alpaca_sel, alpaca_tokens = _take_until_tokens(
                alpaca, desired_alpaca_tokens
            )
        teacher_sel = teacher

    alpaca_sel, teacher_sel = _cap_pool(
        alpaca_sel, teacher_sel, cfg.target_samples
    )
    alpaca_tokens = sum(int(r["answer_tokens"]) for r in alpaca_sel)
    teacher_tokens = sum(int(r["answer_tokens"]) for r in teacher_sel)
    total_tokens = alpaca_tokens + teacher_tokens
    actual_ratio = teacher_tokens / total_tokens if total_tokens else 0.0

    mixed = alpaca_sel + teacher_sel
    rng.shuffle(mixed)
    train_path = out_dir / "train.parquet"
    _write_parquet(mixed, train_path)

    control_path = None
    control_rows = 0
    if cfg.write_alpaca_control:
        n_control = min(len(mixed), len(alpaca))
        control = alpaca[:n_control]
        control_path = out_dir / "alpaca_control.parquet"
        _write_parquet(control, control_path)
        control_rows = len(control)

    baseline_budget = None
    budget_path = Path(cfg.baseline_budget)
    if budget_path.exists():
        baseline_budget = json.loads(budget_path.read_text(encoding="utf-8"))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config": cfg.to_dict(),
        "filtered": {
            "path": str(filtered_path),
            "sha256": _sha256(filtered_path),
            "rows": len(teacher_raw),
            "usable_rows": len(teacher),
        },
        "alpaca": {
            "files": _file_info(alpaca_paths),
            "rows": len(alpaca),
            "answer_tokens_available": alpaca_tokens_total,
        },
        "mix": {
            "requested_teacher_ratio": ratio,
            "actual_teacher_ratio": round(actual_ratio, 4),
            "samples": {
                "alpaca": len(alpaca_sel),
                "teacher": len(teacher_sel),
                "total": len(mixed),
            },
            "answer_tokens": {
                "alpaca": alpaca_tokens,
                "teacher": teacher_tokens,
                "total": total_tokens,
            },
            "avg_answer_tokens": {
                "alpaca": round(
                    alpaca_tokens / len(alpaca_sel), 2
                ) if alpaca_sel else 0.0,
                "teacher": round(
                    teacher_tokens / len(teacher_sel), 2
                ) if teacher_sel else 0.0,
            },
        },
        "outputs": {
            "train": str(train_path),
            "alpaca_control": str(control_path) if control_path else None,
            "alpaca_control_rows": control_rows,
        },
        "baseline_budget": baseline_budget,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"dataset: {len(mixed):,} rows "
        f"(alpaca={len(alpaca_sel):,}, teacher={len(teacher_sel):,}) -> {train_path}"
    )
    print(
        f"  answer tokens: alpaca={alpaca_tokens:,}, teacher={teacher_tokens:,}, "
        f"teacher_ratio={actual_ratio:.3f} (requested {ratio:.3f})"
    )
    if control_path:
        print(f"  alpaca control: {control_rows:,} rows -> {control_path}")
    print(f"  manifest: {manifest_path}")
    return manifest
