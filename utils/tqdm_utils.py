"""Terminal-only progress bars and the corresponding run log.

``run_pipeline.sh`` pipes stdout/stderr through ``tee`` so that ordinary
output is shown in the terminal and stored in the log file.  tqdm is pointed
at the controlling terminal instead, which keeps progress bars out of that
pipe.  Messages printed while a bar is active go through ``pbar.write`` on the
same terminal stream, so tqdm can clear the bar, emit the line, and redraw the
bar below it in a deterministic order.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, TextIO

from tqdm import tqdm

PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[{elapsed} elapsed, {remaining} left]"
)


@lru_cache(maxsize=1)
def terminal_stream() -> TextIO | None:
    """Return the controlling terminal, or None when there is no TTY."""
    try:
        return open("/dev/tty", "w", encoding="utf-8", buffering=1)
    except OSError:
        return None


def progress_kwargs(*, disabled: bool = False) -> dict[str, Any]:
    """tqdm options that keep the progress bar off stdout/stderr."""
    stream = terminal_stream()
    if disabled or stream is None:
        return {"disable": True}
    return {"file": stream, "disable": False}


def log_line(message: str) -> None:
    """Append one line to the log file selected by run_pipeline.sh."""
    path = os.environ.get("QUIN_LOG_FILE")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def progress_write(pbar: tqdm, message: str) -> None:
    """Write a log line above the active bar and into the run log.

    When tqdm is disabled there is no terminal stream; a normal print is
    captured by the pipeline's ``tee`` for both the terminal and the log.
    """
    stream = getattr(pbar, "fp", None)
    if getattr(pbar, "disable", False) or stream is None:
        print(message, flush=True)
        return
    pbar.write(message, file=stream)
    log_line(message)
