"""Generate or verify the frozen evaluation-set manifest.

The manifest pins every JSONL file by SHA256 so that a later run can prove it
evaluated against the same data.  Re-run with ``--check`` in CI or before an
expensive evaluation; any changed hash exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
REQUIRED_FIELDS = ("id", "category", "split", "instruction", "check")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def build_manifest(eval_dir: str | Path) -> dict[str, Any]:
    root = Path(eval_dir)
    files = sorted(p for p in root.glob("*.jsonl"))
    if not files:
        raise ValueError(f"no evaluation JSONL files under {root}")

    seen_ids: set[str] = set()
    files_meta: list[dict[str, Any]] = []
    all_categories: Counter[str] = Counter()
    all_splits: Counter[str] = Counter()
    total = 0

    for path in files:
        rows = _read_jsonl(path)
        for row_no, row in enumerate(rows, start=1):
            for field in REQUIRED_FIELDS:
                if not str(row.get(field, "")).strip():
                    raise ValueError(
                        f"{path}:{row_no}: missing required field {field!r}"
                    )
            sample_id = str(row["id"])
            if sample_id in seen_ids:
                raise ValueError(f"duplicate evaluation id: {sample_id!r}")
            seen_ids.add(sample_id)

        categories = Counter(str(r["category"]) for r in rows)
        splits = Counter(str(r["split"]) for r in rows)
        checks = Counter(str(r["check"]) for r in rows)
        all_categories.update(categories)
        all_splits.update(splits)
        total += len(rows)
        files_meta.append(
            {
                "file": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "samples": len(rows),
                "categories": dict(sorted(categories.items())),
                "splits": dict(sorted(splits.items())),
                "checks": dict(sorted(checks.items())),
            }
        )

    return {
        "version": 1,
        "files": files_meta,
        "total_samples": total,
        "unique_ids": len(seen_ids),
        "categories": dict(sorted(all_categories.items())),
        "splits": dict(sorted(all_splits.items())),
    }


def verify_manifest(eval_dir: str | Path, manifest_path: str | Path) -> list[str]:
    expected = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = build_manifest(eval_dir)
    problems: list[str] = []

    expected_files = {entry["file"]: entry for entry in expected.get("files", [])}
    actual_files = {entry["file"]: entry for entry in actual["files"]}
    for name in sorted(set(expected_files) | set(actual_files)):
        if name not in actual_files:
            problems.append(f"missing file: {name}")
        elif name not in expected_files:
            problems.append(f"unexpected file: {name}")
        elif expected_files[name]["sha256"] != actual_files[name]["sha256"]:
            problems.append(f"hash mismatch: {name}")
    if expected.get("total_samples") != actual["total_samples"]:
        problems.append(
            f"total_samples mismatch: {expected.get('total_samples')} "
            f"!= {actual['total_samples']}"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="evaluation/datasets",
        help="directory containing the frozen JSONL files",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=f"manifest path (default: <dir>/{MANIFEST_NAME})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify against the existing manifest instead of writing it",
    )
    args = parser.parse_args()

    root = Path(args.dir)
    out = Path(args.out) if args.out else root / MANIFEST_NAME
    if args.check:
        if not out.exists():
            print(f"manifest not found: {out}")
            return 1
        problems = verify_manifest(root, out)
        if problems:
            for problem in problems:
                print(f"FAIL: {problem}")
            return 1
        print(f"OK: {out} matches {root}")
        return 0

    manifest = build_manifest(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {out}: {manifest['total_samples']} samples, "
        f"{len(manifest['files'])} file(s)"
    )
    for entry in manifest["files"]:
        print(f"  {entry['file']:36s} {entry['samples']:3d}  {entry['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
