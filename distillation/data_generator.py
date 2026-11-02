"""Run teacher tasks concurrently and persist every raw response.

Resume is request-ID based: a task is skipped only when an earlier record for
the same request ID has ``status == "ok"``.  Error records stay in the file
and are retried on the next run.

Writes happen on the main thread as futures complete, so the raw JSONL stays
valid even if a worker fails mid-run.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from config import DistillGenerationConfig, TeacherConfig
from utils.tqdm_utils import PROGRESS_BAR_FORMAT, progress_kwargs

from .prompts import PROMPT_VERSION, build_messages
from .task_builder import load_tasks, run_task_builder
from .teacher_client import TeacherClient
from .text_utils import canonical_json, hash_text


@dataclass
class ResumeState:
    done: set[str]
    records: list[dict[str, Any]]
    prompt_tokens: int
    completion_tokens: int


def _read_raw(path: Path) -> ResumeState:
    done: set[str] = set()
    records: list[dict[str, Any]] = []
    prompt_tokens = 0
    completion_tokens = 0
    if not path.exists():
        return ResumeState(done, records, prompt_tokens, completion_tokens)

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave one partial trailing line.  Earlier lines
                # are still valid and should be reused.
                print(f"warning: ignoring malformed raw record at {path}:{line_no}")
                continue
            records.append(record)
            usage = record.get("usage") or {}
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)
            if record.get("status") == "ok" and record.get("content"):
                done.add(str(record.get("request_id", "")))
    return ResumeState(done, records, prompt_tokens, completion_tokens)


def _request_identity(task: dict[str, Any], messages: list[dict[str, str]]) -> str:
    """Include the concrete prompt so template edits invalidate cached rows."""
    prompt_hash = hash_text(canonical_json(messages))[:16]
    return f"{task['task_id']}#{str(task.get('task_hash', ''))[:16]}#{prompt_hash}"


def _interleave_by_category(
    items: list[tuple[dict[str, Any], list[dict[str, str]], str]],
) -> list[tuple[dict[str, Any], list[dict[str, str]], str]]:
    """Round-robin tasks across categories.

    Task building groups by category, so a plain ``[:limit]`` would spend a
    small smoke-test budget entirely on the first category.
    """
    buckets: dict[str, list[tuple[dict[str, Any], list[dict[str, str]], str]]] = {}
    order: list[str] = []
    for item in items:
        category = str(item[0].get("category", ""))
        if category not in buckets:
            buckets[category] = []
            order.append(category)
        buckets[category].append(item)
    out: list[tuple[dict[str, Any], list[dict[str, str]], str]] = []
    while any(buckets[category] for category in order):
        for category in order:
            if buckets[category]:
                out.append(buckets[category].pop(0))
    return out


def _record_for_task(
    task: dict[str, Any],
    messages: list[dict[str, str]],
    request_id: str,
    model: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "task_id": task["task_id"],
        "task_hash": task.get("task_hash", ""),
        "category": task.get("category", ""),
        "difficulty": task.get("difficulty", ""),
        "seed": task.get("seed", {}),
        "seed_source": task.get("source", ""),
        "prompt_version": task.get("prompt_version", PROMPT_VERSION),
        "model": model,
        "params": params,
        "messages": messages,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _run_one(
    client: TeacherClient,
    task: dict[str, Any],
    messages: list[dict[str, str]],
    request_id: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    record = _record_for_task(task, messages, request_id, client.model, params)
    started = time.time()
    try:
        data = client.chat(messages)
        record["status"] = "ok"
        record["content"] = client.extract_content(data)
        record["finish_reason"] = client.finish_reason(data)
        record["usage"] = client.usage(data)
        record["response"] = data
    except Exception as exc:  # noqa: BLE001 - recorded and retried next run
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)[:1000]
        record["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    record["elapsed_sec"] = round(time.time() - started, 3)
    return record


def _cost(usage: dict[str, int], cfg: TeacherConfig) -> float:
    return (
        usage.get("prompt_tokens", 0) / 1000.0 * float(cfg.price_input_per_1k)
        + usage.get("completion_tokens", 0) / 1000.0
        * float(cfg.price_output_per_1k)
    )


def _sum_usage(records: list[dict[str, Any]]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for record in records:
        usage = record.get("usage") or {}
        for key in total:
            total[key] += int(usage.get(key) or 0)
    return total


def _merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    """Add several usage dictionaries (e.g. resumed records + this run)."""
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in total:
            total[key] += int(usage.get(key) or 0)
    return total


def _print_prompts(tasks: list[dict[str, Any]], limit: int | None) -> None:
    selected = tasks if limit is None else tasks[:limit]
    for task in selected:
        messages = build_messages(task)
        print("=" * 72)
        print(f"{task['task_id']}  category={task['category']}  "
              f"difficulty={task['difficulty']}")
        for message in messages:
            print(f"--- {message['role']} ---")
            print(message["content"])
    print("=" * 72)
    print(f"dry run: {len(selected):,} prompt(s), no API requests sent")


def run_generation(
    cfg: DistillGenerationConfig,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    reuse_tasks: bool = False,
    categories: list[str] | None = None,
) -> Path | None:
    task_path = Path(cfg.tasks_dir) / f"{cfg.run_id}.jsonl"
    if reuse_tasks and task_path.exists():
        tasks = load_tasks(task_path)
        print(f"tasks: reusing {len(tasks):,} from {task_path}")
    else:
        task_path = run_task_builder(cfg)
        tasks = load_tasks(task_path)

    if categories:
        allowed = set(categories)
        unknown = allowed - {t.get("category") for t in tasks}
        if unknown:
            raise ValueError(f"unknown categories: {sorted(unknown)}")
        tasks = [t for t in tasks if t.get("category") in allowed]
        print(f"tasks: {len(tasks):,} after category filter {sorted(allowed)}")

    if dry_run:
        _print_prompts(tasks, limit)
        return None

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{cfg.run_id}.jsonl"

    state = _read_raw(raw_path)
    if state.done:
        print(f"resume: {len(state.done):,} completed request(s) in {raw_path}")

    teacher_cfg = TeacherConfig.from_yaml(cfg.teacher_config)
    client = TeacherClient(teacher_cfg)
    params = client.generation_params()
    print(f"teacher: {client.model} at {client.base_url}")
    print(f"  params: {json.dumps(params, ensure_ascii=False, sort_keys=True)}")

    pending: list[tuple[dict[str, Any], list[dict[str, str]], str]] = []
    for task in tasks:
        messages = build_messages(task)
        identity = _request_identity(task, messages)
        request_id = client.request_id(identity, cfg.prompt_version)
        if request_id in state.done:
            continue
        pending.append((task, messages, request_id))

    if limit is not None:
        pending = _interleave_by_category(pending)[: max(0, int(limit))]
    if cfg.max_requests is not None:
        remaining = max(0, int(cfg.max_requests))
        pending = pending[:remaining]

    if cfg.max_completion_tokens is not None:
        used = state.completion_tokens
        if used >= int(cfg.max_completion_tokens):
            print(
                f"budget: completion tokens already at {used:,} "
                f"(limit {int(cfg.max_completion_tokens):,}); nothing to do"
            )
            pending = []

    if not pending:
        print("generation: nothing to do")
        return raw_path

    print(f"generation: {len(pending):,} pending request(s), "
          f"concurrency={teacher_cfg.concurrency}")
    started = time.time()
    errors = 0
    written = 0
    new_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    max_workers = max(1, int(teacher_cfg.concurrency))
    completion_budget = cfg.max_completion_tokens
    budget_exhausted = False
    pending_iter = iter(pending)

    with open(raw_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(
        max_workers=max_workers
    ) as pool:
        futures: dict[Future, tuple[dict[str, Any], list[dict[str, str]], str]] = {}

        def submit_next() -> bool:
            try:
                item = next(pending_iter)
            except StopIteration:
                return False
            task, messages, request_id = item
            futures[pool.submit(_run_one, client, task, messages, request_id, params)] = item
            return True

        # Keep a small queue in flight, but stop submitting as soon as the
        # completion-token budget is reached.  Running requests still finish,
        # so the overshoot is bounded by the queue size.
        for _ in range(max_workers * 2):
            if not submit_next():
                break

        with tqdm(
            total=len(pending),
            desc="teacher",
            unit="req",
            dynamic_ncols=True,
            bar_format=PROGRESS_BAR_FORMAT,
            **progress_kwargs(),
        ) as pbar:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    record = future.result()
                    del futures[future]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    written += 1
                    if record["status"] != "ok":
                        errors += 1
                    usage = record.get("usage") or {}
                    for key in new_usage:
                        new_usage[key] += int(usage.get(key) or 0)
                    pbar.update(1)
                    pbar.set_postfix(errors=errors)

                    if (
                        completion_budget is not None
                        and state.completion_tokens
                        + new_usage["completion_tokens"]
                        >= int(completion_budget)
                    ):
                        budget_exhausted = True
                    if not budget_exhausted:
                        submit_next()
                    elif not pbar.total == written + len(futures):
                        # The budget stopped new submissions; keep the bar honest.
                        pbar.total = written + len(futures)

    elapsed = time.time() - started
    total_usage = _merge_usage(_sum_usage(state.records), new_usage)
    total_records = len(state.records) + written
    stats = {
        "run_id": cfg.run_id,
        "raw_path": str(raw_path),
        "teacher": client.describe(),
        "requests_written": written,
        "errors_this_run": errors,
        "requests_total": total_records,
        "usage_total": total_usage,
        "usage_this_run": new_usage,
        "estimated_cost": _cost(total_usage, teacher_cfg),
        "elapsed_sec": round(elapsed, 2),
        "requests_per_minute": round(written / elapsed * 60, 2) if elapsed else 0.0,
    }
    stats_path = out_dir / f"{cfg.run_id}.stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"generation: wrote {written:,} record(s), errors={errors:,}, "
        f"elapsed={elapsed:.1f}s"
    )
    print(
        f"  tokens: prompt={total_usage['prompt_tokens']:,}, "
        f"completion={total_usage['completion_tokens']:,}, "
        f"estimated_cost={stats['estimated_cost']:.4f}"
    )
    print(f"  stats: {stats_path}")
    if errors:
        print("  rerun the same command to retry failed requests")
    return raw_path
