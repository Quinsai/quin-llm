"""Compare baseline and candidate evaluation outputs.

Both inputs are the raw JSONL files written by ``evaluation.run_eval``.  Only
examples present in both files are paired.  Accuracy is computed on the paired
subset; improvement confidence intervals come from a paired bootstrap and
McNemar's test is used for the binary correct/incorrect signal.

Example:
    python -m evaluation.compare_models \
        --baseline reports/eval_baseline.jsonl \
        --candidate reports/eval_mixed.jsonl \
        --report reports/distill_report.md
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import mcnemar


def _load(path: str | Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", "")).strip()
            if not sample_id:
                raise ValueError(f"{path}:{line_no}: missing id")
            if sample_id in records:
                raise ValueError(f"{path}: duplicate id {sample_id!r}")
            records[sample_id] = record
    return records


def _paired(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ids = sorted(set(baseline) & set(candidate))
    return [(baseline[i], candidate[i]) for i in ids]


def _bootstrap_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    field: str,
    n_boot: int,
    seed: int,
) -> list[float] | None:
    values = [
        (float(b[field]), float(c[field]))
        for b, c in pairs
        if b.get(field) is not None and c.get(field) is not None
    ]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    deltas = []
    n = len(values)
    for _ in range(max(1, int(n_boot))):
        b_sum = 0.0
        c_sum = 0.0
        for _ in range(n):
            b, c = values[rng.randrange(n)]
            b_sum += b
            c_sum += c
        deltas.append((c_sum - b_sum) / n)
    deltas.sort()
    lo = deltas[int(0.025 * len(deltas))]
    hi = deltas[min(len(deltas) - 1, int(0.975 * len(deltas)))]
    return [round(lo, 4), round(hi, 4)]


def _summary(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    binary = [
        (bool(b["correct"]), bool(c["correct"]))
        for b, c in pairs
        if b.get("correct") is not None and c.get("correct") is not None
    ]
    base_acc = sum(b for b, _ in binary) / len(binary) if binary else None
    cand_acc = sum(c for _, c in binary) / len(binary) if binary else None
    scores = [
        (float(b["score"]), float(c["score"]))
        for b, c in pairs
        if b.get("score") is not None and c.get("score") is not None
    ]
    base_score = sum(b for b, _ in scores) / len(scores) if scores else None
    cand_score = sum(c for _, c in scores) / len(scores) if scores else None
    return {
        "n_pairs": len(pairs),
        "n_binary": len(binary),
        "n_scored": len(scores),
        "baseline_accuracy": round(base_acc, 4) if base_acc is not None else None,
        "candidate_accuracy": round(cand_acc, 4) if cand_acc is not None else None,
        "delta_accuracy": (
            round(cand_acc - base_acc, 4)
            if base_acc is not None and cand_acc is not None
            else None
        ),
        "accuracy_delta_ci95": _bootstrap_delta(
            pairs, field="correct", n_boot=n_boot, seed=seed
        ),
        "baseline_score": round(base_score, 4) if base_score is not None else None,
        "candidate_score": round(cand_score, 4) if cand_score is not None else None,
        "delta_score": (
            round(cand_score - base_score, 4)
            if base_score is not None and cand_score is not None
            else None
        ),
        "score_delta_ci95": _bootstrap_delta(
            pairs, field="score", n_boot=n_boot, seed=seed
        ),
        "mcnemar": mcnemar(
            [b.get("correct") for b, _ in pairs],
            [c.get("correct") for _, c in pairs],
        ),
    }


def _group(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], *fields: str
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for b, c in pairs:
        key = "/".join(str(b.get(field, "")) for field in fields)
        groups[key].append((b, c))
    return dict(sorted(groups.items()))


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_ci(ci: list[float] | None) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def _cell(text: Any, limit: int = 160) -> str:
    value = str(text if text is not None else "").replace("\n", " ").replace("|", "\\|")
    return value if len(value) <= limit else value[: limit - 1] + "…"


def _table(
    title: str,
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    *,
    n_boot: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    lines = [
        f"### {title}",
        "",
        "| Group | n | Baseline acc | Mixed acc | Δ acc | 95% CI | McNemar p | Baseline score | Mixed score | Δ score |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    metrics: dict[str, Any] = {}
    for name, pairs in groups.items():
        summary = _summary(pairs, n_boot=n_boot, seed=seed)
        metrics[name] = summary
        p = summary["mcnemar"]["p_value"] if summary["mcnemar"] else None
        lines.append(
            "| {name} | {n} | {ba} | {ca} | {da} | {ci} | {p} | {bs} | {cs} | {ds} |".format(
                name=_cell(name),
                n=summary["n_binary"],
                ba=_fmt(summary["baseline_accuracy"]),
                ca=_fmt(summary["candidate_accuracy"]),
                da=(
                    f"{summary['delta_accuracy']:+.3f}"
                    if summary["delta_accuracy"] is not None
                    else "—"
                ),
                ci=_fmt_ci(summary["accuracy_delta_ci95"]),
                p=_fmt(p, 4),
                bs=_fmt(summary["baseline_score"]),
                cs=_fmt(summary["candidate_score"]),
                ds=(
                    f"{summary['delta_score']:+.3f}"
                    if summary["delta_score"] is not None
                    else "—"
                ),
            )
        )
    lines.append("")
    return "\n".join(lines), metrics


def _cases(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    candidate_wins: bool,
    limit: int,
) -> list[str]:
    selected = []
    for baseline, candidate in pairs:
        if baseline.get("correct") is None or candidate.get("correct") is None:
            continue
        if candidate_wins:
            match = bool(candidate["correct"]) and not bool(baseline["correct"])
        else:
            match = bool(baseline["correct"]) and not bool(candidate["correct"])
        if match:
            selected.append((baseline, candidate))
        if len(selected) >= limit:
            break
    lines = []
    for baseline, candidate in selected:
        lines.extend(
            [
                f"- **{candidate.get('category')}/{candidate.get('split')} · {candidate.get('id')}**",
                f"  - Q: {_cell(candidate.get('instruction'))}",
                f"  - Ref: {_cell(candidate.get('reference'))}",
                f"  - Baseline: {_cell(baseline.get('prediction'))}",
                f"  - Mixed: {_cell(candidate.get('prediction'))}",
            ]
        )
    if not lines:
        lines.append("- 无")
    return lines


def run_compare(
    baseline_path: str,
    candidate_path: str,
    report_path: str,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    max_cases: int = 5,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    pairs = _paired(baseline, candidate)
    if not pairs:
        raise ValueError("baseline and candidate have no evaluation ids in common")

    overall = _summary(pairs, n_boot=n_boot, seed=seed)
    category_text, category_metrics = _table(
        "分类别", _group(pairs, "category"), n_boot=n_boot, seed=seed
    )
    split_text, split_metrics = _table(
        "分拆分", _group(pairs, "split"), n_boot=n_boot, seed=seed
    )
    pair_text, pair_metrics = _table(
        "类别 × 拆分",
        _group(pairs, "category", "split"),
        n_boot=n_boot,
        seed=seed,
    )

    unscored = [
        (b, c)
        for b, c in pairs
        if b.get("correct") is None or c.get("correct") is None
    ]
    retention_cases = []
    for base_row, cand_row in unscored[:3]:
        retention_cases.extend(
            [
                f"- **{cand_row.get('id')}** — {_cell(cand_row.get('instruction'))}",
                f"  - Baseline: {_cell(base_row.get('prediction'), 300)}",
                f"  - Mixed: {_cell(cand_row.get('prediction'), 300)}",
            ]
        )
    if not retention_cases:
        retention_cases.append("- 无未判分样本")

    baseline_steps = sorted(
        {str(r.get("checkpoint", "")) for r in baseline.values() if r.get("checkpoint")}
    )
    candidate_steps = sorted(
        {str(r.get("checkpoint", "")) for r in candidate.values() if r.get("checkpoint")}
    )
    report = [
        "# Teacher 蒸馏 SFT 对比报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 运行",
        "",
        f"- Baseline：`{baseline_path}`（{', '.join(baseline_steps) or 'unknown'}）",
        f"- Mixed：`{candidate_path}`（{', '.join(candidate_steps) or 'unknown'}）",
        f"- 配对样本：{len(pairs):,}（Binary {overall['n_binary']:,}，"
        f"Scored {overall['n_scored']:,}）",
        f"- Baseline-only ids：{len(set(baseline) - set(candidate)):,}；"
        f"Mixed-only ids：{len(set(candidate) - set(baseline)):,}",
        f"- Bootstrap：{n_boot:,}，seed={seed}",
        "",
        "## 总体",
        "",
        f"- Baseline accuracy：{_fmt(overall['baseline_accuracy'])}；"
        f"Mixed accuracy：{_fmt(overall['candidate_accuracy'])}；"
        f"Δ = {overall['delta_accuracy']:+.3f}"
        if overall["delta_accuracy"] is not None
        else "- accuracy：无 binary 样本",
        f"- Δ accuracy 95% CI：{_fmt_ci(overall['accuracy_delta_ci95'])}",
        f"- McNemar p：{_fmt(overall['mcnemar']['p_value'] if overall['mcnemar'] else None, 4)}",
        f"- Baseline mean score：{_fmt(overall['baseline_score'])}；"
        f"Mixed mean score：{_fmt(overall['candidate_score'])}；"
        f"Δ = {overall['delta_score']:+.3f}"
        if overall["delta_score"] is not None
        else "- score：无 scored 样本",
        "",
        category_text,
        split_text,
        pair_text,
        "## 代表性变化",
        "",
        "### Mixed 变好",
        "",
        *_cases(pairs, candidate_wins=True, limit=max_cases),
        "",
        "### Mixed 变差（遗忘/回归）",
        "",
        *_cases(pairs, candidate_wins=False, limit=max_cases),
        "",
        "## 未自动判分的开放题（retention）",
        "",
        f"共 {len(unscored):,} 条。以下仅展示输出，quality 需人工或独立 judge 评分。",
        "",
        *retention_cases,
        "",
        "## 说明与限制",
        "",
        "- 规则判分优先；alias 匹配对短答案较稳，但对长句和否定句可能偏宽松。",
        "- 开放题未自动评分，retention 结论需结合人工检查。",
        "- 只有配对样本参与比较；缺失 id 不会补 0 分。",
        "- 若两组训练步数、数据量或超参不同，本报告不能单独证明教师数据的作用。",
        "",
    ]
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(report), encoding="utf-8")

    metrics = {
        "baseline": baseline_path,
        "candidate": candidate_path,
        "paired": len(pairs),
        "overall": overall,
        "by_category": category_metrics,
        "by_split": split_metrics,
        "by_category_split": pair_metrics,
        "unscored": len(unscored),
        "report": str(report_file),
    }
    print(f"compare: {len(pairs):,} paired examples -> {report_file}")
    print(
        f"  accuracy: baseline={_fmt(overall['baseline_accuracy'])} "
        f"mixed={_fmt(overall['candidate_accuracy'])} "
        f"delta={overall['delta_accuracy']}"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and candidate evals")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    metrics = run_compare(
        args.baseline,
        args.candidate,
        args.report,
        n_boot=args.bootstrap,
        seed=args.seed,
        max_cases=args.max_cases,
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  metrics: {out}")


if __name__ == "__main__":
    main()
