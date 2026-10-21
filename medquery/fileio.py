"""Crash-safe output helpers shared by the ingest build and (later) the eval runner."""

import os
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


class LockHeld(Exception):
    """Another run holds the lock (or a crashed run left it behind)."""


@contextmanager
def atomic_write(path: Path) -> Iterator[TextIO]:
    """Write text to `path` via a temp file in the same folder, then os.replace() it in.

    Readers see either the old file or the complete new one, never a partial write. On error
    the temp file is removed and `path` is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Binary counterpart of atomic_write: temp file in the same folder, then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold `path` as a lock file for the duration of the block.

    Creation is exclusive (O_CREAT | O_EXCL), so two runs cannot both acquire it. The lock is
    removed on exit, including on error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = path.read_text(encoding="utf-8", errors="replace").strip() or "unknown"
        raise LockHeld(
            f"{path} exists ({holder}). Another run is in progress. If no run is active, a "
            f"previous run crashed: delete {path.name} and retry."
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(f"pid {os.getpid()}, started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)
