"""The git commit and dirty-tree flag recorded in every report."""

import subprocess
from pathlib import Path

from medquery import paths


def git_state(root: Path = paths.ROOT) -> dict:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=30, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "dirty": None if status is None else bool(status),
    }
