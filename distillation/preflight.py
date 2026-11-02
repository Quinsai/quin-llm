"""Pre-flight checks for the distillation experiment.

Run this on the training server before spending API budget or GPU time.  It
checks the environment, data/checkpoint paths, teacher credentials, and computes
the baseline SFT step budget used by the controlled comparison.

Examples:
    python -m distillation.preflight
    python -m distillation.preflight --ping --json-out reports/preflight.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import importlib.util
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import TeacherConfig, load_yaml

from .task_builder import assert_disjoint_entities
from .teacher_client import TeacherClient, api_key_source, resolve_api_key


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


_ENV_REF_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _expand_env(value: str | None) -> str:
    """Expand ``${VAR}`` and treat unresolved references as empty.

    ``os.path.expandvars`` leaves unknown references untouched, which would
    make an unset ``TEACHER_BASE_URL`` look non-empty to preflight.
    """
    expanded = os.path.expandvars(value or "")
    return "" if _ENV_REF_RE.search(expanded) else expanded


def _glob(pattern: str) -> list[str]:
    import glob

    return sorted(glob.glob(pattern))


def _sft_log_step(path: Path) -> int | None:
    if not path.exists():
        return None
    last = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                step = int(float(row.get("step") or 0))
            except (TypeError, ValueError):
                continue
            if step > 0:
                last = step if last is None else max(last, step)
    return last


def compute_baseline_budget(
    meta_path: Path,
    log_path: Path,
    sft_config_path: Path,
    *,
    batch_size: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_cfg: dict[str, Any] = {}
    if sft_config_path.exists():
        train_cfg = load_yaml(sft_config_path).get("train", {}) or {}

    batch_size = int(batch_size or train_cfg.get("batch_size") or 32)
    epochs = int(epochs or train_cfg.get("epochs") or 2)
    replay_prob = 0.0
    if sft_config_path.exists():
        replay_prob = float(
            (load_yaml(sft_config_path).get("data", {}) or {}).get(
                "replay_prob", 0.0
            )
        )

    train_samples = int(meta.get("train_samples") or 0)
    answer_tokens = int(meta.get("train_answer_tokens") or 0)
    planned_steps = (
        math.ceil(train_samples / batch_size) * epochs if train_samples else 0
    )
    log_step = _sft_log_step(log_path)
    configured_steps = train_cfg.get("max_steps")
    if configured_steps:
        max_steps = int(configured_steps)
    elif log_step:
        max_steps = int(log_step)
    else:
        max_steps = planned_steps

    avg_answer = answer_tokens / train_samples if train_samples else 0.0
    answer_tokens_seen = (
        max_steps * batch_size * avg_answer * (1.0 - replay_prob)
        if max_steps
        else 0.0
    )
    return {
        "sft_meta": str(meta_path),
        "sft_log": str(log_path) if log_path.exists() else None,
        "sft_config": str(sft_config_path) if sft_config_path.exists() else None,
        "train_samples": train_samples,
        "train_answer_tokens": answer_tokens,
        "avg_answer_tokens": round(avg_answer, 3),
        "batch_size": batch_size,
        "epochs": epochs,
        "replay_prob": replay_prob,
        "planned_steps": planned_steps,
        "log_final_step": log_step,
        "max_steps": max_steps,
        "answer_tokens_seen_estimate": int(round(answer_tokens_seen)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class Checklist:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str, *, required: bool = True) -> None:
        status = "PASS" if ok else ("FAIL" if required else "WARN")
        self.items.append(
            {"name": name, "status": status, "detail": detail, "required": required}
        )

    def fail(self, name: str, detail: str, *, required: bool = True) -> None:
        self.add(name, False, detail, required=required)

    def print(self) -> None:
        for item in self.items:
            print(f"[{item['status']}] {item['name']}: {item['detail']}")

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item["status"] == "FAIL"]


def _check_packages(check: Checklist) -> None:
    for package in ("requests", "pandas", "pyarrow", "datasets", "torch"):
        spec = importlib.util.find_spec(package)
        version = _version(package)
        check.add(
            f"package:{package}",
            spec is not None,
            f"version={version or 'missing'}",
            required=package in {"requests", "torch"},
        )


def _check_gpu(check: Checklist) -> None:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        check.fail("torch import", str(exc), required=True)
        return
    available = torch.cuda.is_available()
    detail = (
        f"cuda={available}, count={torch.cuda.device_count()}, "
        f"name={torch.cuda.get_device_name(0) if available else 'n/a'}"
    )
    check.add("cuda", available, detail, required=False)


def _check_teacher(
    check: Checklist,
    teacher_config: str,
    *,
    ping: bool,
) -> dict[str, Any] | None:
    path = Path(teacher_config)
    if not path.exists():
        check.fail("teacher config", f"missing {path}")
        return None
    cfg = TeacherConfig.from_yaml(path)
    base_url = _expand_env(cfg.base_url)
    model = _expand_env(cfg.model)
    source = api_key_source(cfg)
    key_error = None
    try:
        api_key = resolve_api_key(cfg)
    except ValueError as exc:
        api_key = ""
        key_error = str(exc)
    check.add("teacher base_url", bool(base_url), base_url or "empty", required=False)
    check.add("teacher model", bool(model), model or "empty", required=False)
    check.add(
        f"teacher api key ({source})",
        bool(api_key) or not (cfg.api_key or cfg.api_key_file or cfg.api_key_env),
        "present" if api_key else (key_error or "missing"),
        required=False,
    )
    info = {
        "base_url": base_url,
        "model": model,
        "api_key_present": bool(api_key),
        "api_key_source": source,
    }
    if ping:
        try:
            client = TeacherClient(cfg)
            reply = client.ping()
            check.add("teacher ping", bool(reply), repr(reply))
        except Exception as exc:  # noqa: BLE001
            check.fail("teacher ping", str(exc), required=False)
    return info


def _check_paths(check: Checklist, eval_dir: str) -> dict[str, Any]:
    tokenizer = Path("processed_data/tokenizer/tokenizer.pkl")
    pretrain = Path("checkpoints/pretrain_full/best.pt")
    alpaca = _glob("data/raw/alpaca/*.parquet")
    pretrain_meta = Path("processed_data/pretrain/meta.json")
    sft_meta = Path("processed_data/sft/meta.json")
    sft_log = Path("checkpoints/sft_full/sft_log.csv")
    sft_cfg = Path("configs/sft_full.yaml")

    check.add("tokenizer", tokenizer.exists(), str(tokenizer))
    check.add("pretrain checkpoint", pretrain.exists(), str(pretrain))
    check.add(
        "alpaca parquet",
        bool(alpaca),
        f"{len(alpaca)} file(s)" if alpaca else "data/raw/alpaca/*.parquet",
    )
    check.add(
        "pretrain meta", pretrain_meta.exists(), str(pretrain_meta), required=False
    )
    check.add("sft meta", sft_meta.exists(), str(sft_meta), required=False)
    check.add("sft log", sft_log.exists(), str(sft_log), required=False)
    check.add("sft config", sft_cfg.exists(), str(sft_cfg), required=False)
    check.add(
        "evaluation datasets",
        Path(eval_dir).exists(),
        str(eval_dir),
        required=False,
    )

    train_seeds = Path("distillation/seeds/train_entities.jsonl")
    eval_seeds = Path("distillation/seeds/eval_entities.jsonl")
    try:
        assert_disjoint_entities(train_seeds, eval_seeds)
        check.add("train/eval entity split", True, "disjoint")
    except Exception as exc:  # noqa: BLE001
        check.fail("train/eval entity split", str(exc))

    for out_dir in (
        "checkpoints/sft_baseline_controlled",
        "checkpoints/sft_distill",
        "data/distill",
    ):
        path = Path(out_dir)
        parent = path if path.exists() else path.parent
        writable = os.access(parent, os.W_OK)
        detail = f"{out_dir} {'exists' if path.exists() else 'not created'}"
        check.add(f"writable:{out_dir}", writable, detail, required=False)
        if path.exists() and path.name.startswith("sft_"):
            check.fail(
                f"output not empty: {out_dir}",
                "remove/rename it before a controlled run; do not overwrite",
                required=False,
            )

    return {
        "tokenizer": str(tokenizer),
        "pretrain_checkpoint": str(pretrain),
        "alpaca_files": alpaca,
        "sft_meta": str(sft_meta),
        "sft_log": str(sft_log),
        "sft_config": str(sft_cfg),
    }


def run_preflight(
    *,
    teacher_config: str = "configs/distill_teacher.yaml",
    eval_dir: str = "evaluation/datasets",
    ping: bool = False,
    budget_out: str = "data/distill/baseline_budget.json",
    batch_size: int | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    check = Checklist()
    _check_packages(check)
    _check_gpu(check)
    paths = _check_paths(check, eval_dir)
    teacher = _check_teacher(check, teacher_config, ping=ping)

    budget = compute_baseline_budget(
        Path(paths["sft_meta"]),
        Path(paths["sft_log"]),
        Path(paths["sft_config"]),
        batch_size=batch_size,
        epochs=epochs,
    )
    if budget is None:
        check.fail(
            "baseline budget",
            f"{paths['sft_meta']} missing; cannot infer controlled max_steps",
            required=False,
        )
    else:
        check.add(
            "baseline budget",
            budget["max_steps"] > 0,
            f"max_steps={budget['max_steps']}, "
            f"answer_tokens_seen~{budget['answer_tokens_seen_estimate']:,}",
        )
        out = Path(budget_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(budget, indent=2), encoding="utf-8")
        print(f"baseline budget -> {out}")

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checks": check.items,
        "paths": paths,
        "teacher": teacher,
        "baseline_budget": budget,
        "failed": len(check.failed),
    }
    check.print()
    if check.failed:
        print(f"\npreflight FAILED: {len(check.failed)} required check(s)")
    else:
        print("\npreflight passed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Distillation pre-flight checks")
    parser.add_argument(
        "--teacher-config", default="configs/distill_teacher.yaml"
    )
    parser.add_argument("--eval-dir", default="evaluation/datasets")
    parser.add_argument("--ping", action="store_true", help="send one tiny API call")
    parser.add_argument(
        "--budget-out", default="data/distill/baseline_budget.json"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="exit 0 even when required checks fail",
    )
    args = parser.parse_args()

    result = run_preflight(
        teacher_config=args.teacher_config,
        eval_dir=args.eval_dir,
        ping=args.ping,
        budget_out=args.budget_out,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"preflight report -> {out}")
    if result["failed"] and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
