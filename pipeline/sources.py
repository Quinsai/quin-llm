"""Helpers for reading local files and HuggingFace datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from datasets import load_dataset


def _load_hf(source: dict):
    return load_dataset(
        source["dataset"],
        source.get("config"),
        split=source.get("split", "train"),
        streaming=source.get("streaming", True),
    )


def iter_source_texts(source: dict) -> Iterator[str]:
    """Yield documents from a pretraining / tokenizer source config."""
    kind = source.get("type", "hf")

    if kind == "file":
        with open(source["path"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.strip():
                    yield line
        return

    if kind == "hf":
        field = source.get("text_field", "text")
        for example in _load_hf(source):
            text = example.get(field, "")
            if text and text.strip():
                yield text
        return

    if kind == "alpaca":
        for example in _load_hf(source):
            parts = [
                example.get("instruction", ""),
                example.get("input", ""),
                example.get("output", ""),
            ]
            text = "\n".join(p.strip() for p in parts if p and p.strip())
            if text:
                yield text
        return

    raise ValueError(f"unknown source type: {kind}")


def parse_sft_file(path: str | Path) -> Iterator[tuple[str, str]]:
    """Parse the project's ``###User:`` / ``###Agent:`` message format.

    Unlike the old implementation, this splits on the ``###User:`` marker rather
    than blank lines, so answers containing blank lines are kept intact.
    """
    text = Path(path).read_text(encoding="utf-8")
    for piece in text.split("###User:")[1:]:
        if "###Agent:" not in piece:
            continue
        user, _, answer = piece.partition("###Agent:")
        user = user.strip("\n")
        answer = answer.strip("\n")
        if user.strip() and answer.strip():
            yield user, answer


def iter_sft_examples(source: dict) -> Iterator[tuple[str, str]]:
    """Yield ``(user_text, answer_text)`` pairs from a SFT source."""
    kind = source.get("type", "hf")

    if kind == "file":
        yield from parse_sft_file(source["path"])
        return

    if kind in ("hf", "alpaca"):
        instruction_field = source.get("instruction_field", "instruction")
        input_field = source.get("input_field", "input")
        output_field = source.get("output_field", "output")
        for example in _load_hf(source):
            instruction = (example.get(instruction_field) or "").strip()
            extra = (example.get(input_field) or "").strip()
            answer = (example.get(output_field) or "").strip()
            if not instruction or not answer:
                continue
            user = instruction + ("\n" + extra if extra else "")
            yield user, answer
        return

    raise ValueError(f"unknown SFT source type: {kind}")
