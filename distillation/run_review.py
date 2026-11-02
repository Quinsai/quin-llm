"""Human-in-the-loop review for filtered teacher data.

Workflow:
1. ``sample`` writes a stratified review CSV (the filter also writes one).
2. A first reviewer fills ``model_verdict`` (keep/reject) and ``model_notes``.
3. The human reviewer fills ``human_verdict`` and ``human_notes``.
4. ``merge`` applies the verdicts; human overrides model.  Rows without a
   verdict keep their automatic status, so ``merge`` is safe to run after the
   first pass and again after the human pass.

Examples:
    python -m distillation.run_review sample \
        --filtered data/distill/filtered/pilot_v1.jsonl \
        --rejected data/distill/filtered/pilot_v1.rejected.jsonl \
        --out data/distill/filtered/pilot_v1.review.csv
    python -m distillation.run_review merge \
        --filtered data/distill/filtered/pilot_v1.jsonl \
        --rejected data/distill/filtered/pilot_v1.rejected.jsonl \
        --review data/distill/filtered/pilot_v1.review.csv \
        --out data/distill/filtered/pilot_v1.reviewed.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .data_filter import write_review_csv
from .text_utils import iter_jsonl


_KEEP = {"keep", "yes", "y", "1", "true", "ok", "accept"}
_REJECT = {"reject", "no", "n", "0", "false", "bad", "drop"}


def _parse_verdict(value: str | None, row_id: str) -> bool | None:
    text = (value or "").strip().lower()
    if not text:
        return None
    if text in _KEEP:
        return True
    if text in _REJECT:
        return False
    raise ValueError(
        f"review row {row_id!r}: unknown verdict {value!r}; "
        "use keep/reject (or leave blank)"
    )


def _load(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(str(path)))


def _write(path: str | Path, records: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cmd_sample(args) -> None:
    kept = _load(args.filtered)
    rejected = _load(args.rejected) if args.rejected else []
    counts = write_review_csv(
        kept,
        rejected,
        args.out,
        fraction=args.fraction,
        min_samples=args.min_samples,
        rejected_fraction=args.rejected_fraction,
        min_rejected=args.min_rejected,
        seed=args.seed,
    )
    print(
        f"review: {counts['kept']:,} kept + {counts['rejected']:,} rejected "
        f"-> {args.out}"
    )


def cmd_merge(args) -> None:
    kept = _load(args.filtered)
    rejected = _load(args.rejected) if args.rejected else []
    with open(args.review, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    review: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = (row.get("sample_id") or "").strip()
        if not sample_id:
            continue
        if sample_id in review:
            raise ValueError(f"duplicate review row for {sample_id!r}")
        review[sample_id] = row

    kept_ids = {str(record.get("id")) for record in kept}
    rejected_ids = {str(record.get("id")) for record in rejected}
    unknown = set(review) - kept_ids - rejected_ids
    if unknown:
        raise ValueError(
            f"review contains {len(unknown):,} unknown sample id(s): "
            f"{sorted(unknown)[:5]}"
        )

    kept_out: list[dict[str, Any]] = []
    rejected_out: list[dict[str, Any]] = []
    stats = {
        "filtered_in": len(kept),
        "rejected_in": len(rejected),
        "review_rows": len(review),
        "model_overrides": 0,
        "human_overrides": 0,
        "conflicts": 0,
        "manual_rejects": 0,
        "rescued": 0,
    }

    for record in kept:
        sample_id = str(record.get("id"))
        row = review.get(sample_id)
        if row is None:
            kept_out.append(record)
            continue
        human = _parse_verdict(row.get("human_verdict"), sample_id)
        model = _parse_verdict(row.get("model_verdict"), sample_id)
        if human is not None:
            stats["human_overrides"] += 1
            verdict, source = human, "human"
        elif model is not None:
            stats["model_overrides"] += 1
            verdict, source = model, "model"
        else:
            kept_out.append(record)
            continue
        if human is not None and model is not None and human != model:
            stats["conflicts"] += 1
        if verdict:
            kept_out.append(record)
        else:
            stats["manual_rejects"] += 1
            updated = dict(record)
            updated["reasons"] = ["manual_reject"]
            updated["details"] = {
                "review_source": source,
                "notes": row.get("human_notes") or row.get("model_notes") or "",
                "auto_reasons": record.get("reasons") or [],
            }
            rejected_out.append(updated)

    for record in rejected:
        sample_id = str(record.get("id"))
        row = review.get(sample_id)
        if row is None:
            rejected_out.append(record)
            continue
        human = _parse_verdict(row.get("human_verdict"), sample_id)
        model = _parse_verdict(row.get("model_verdict"), sample_id)
        if human is not None:
            stats["human_overrides"] += 1
            verdict, source = human, "human"
        elif model is not None:
            stats["model_overrides"] += 1
            verdict, source = model, "model"
        else:
            rejected_out.append(record)
            continue
        if human is not None and model is not None and human != model:
            stats["conflicts"] += 1
        if not verdict:
            rejected_out.append(record)
            continue
        stats["rescued"] += 1
        updated = dict(record)
        auto_reasons = updated.pop("reasons", [])
        updated.pop("details", None)
        updated["review_rescue"] = {
            "review_source": source,
            "notes": row.get("human_notes") or row.get("model_notes") or "",
            "auto_reasons": auto_reasons,
        }
        kept_out.append(updated)

    kept_out.sort(key=lambda r: (str(r.get("category", "")), str(r.get("id", ""))))
    rejected_out.sort(key=lambda r: (str(r.get("category", "")), str(r.get("id", ""))))
    _write(args.out, kept_out)
    _write(args.rejected_out, rejected_out)

    stats["filtered_out"] = len(kept_out)
    stats["rejected_out"] = len(rejected_out)
    stats_path = Path(args.stats_out) if args.stats_out else Path(args.out).with_suffix(".merge.json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"merge: kept {len(kept_out):,}, rejected {len(rejected_out):,} "
        f"(manual_rejects={stats['manual_rejects']:,}, "
        f"rescued={stats['rescued']:,}, conflicts={stats['conflicts']:,})"
    )
    print(f"  stats: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Review teacher data")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="write a stratified review CSV")
    p.add_argument("--filtered", required=True)
    p.add_argument("--rejected", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--min-samples", type=int, default=100)
    p.add_argument("--rejected-fraction", type=float, default=0.02)
    p.add_argument("--min-rejected", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("merge", help="apply model/human verdicts")
    p.add_argument("--filtered", required=True)
    p.add_argument("--rejected", default=None)
    p.add_argument("--review", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rejected-out", required=True)
    p.add_argument("--stats-out", default=None)
    p.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
