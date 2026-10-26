"""Métadonnées jointes à chaque chiffre publié : date, commit, seed, machine."""

from __future__ import annotations

import os
import platform
import resource
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aml.config import SEED


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "inconnu"


def run_meta() -> dict[str, object]:
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--", ".")),
        "seed": SEED,
        "machine": platform.machine(),
        "processor": platform.processor() or "inconnu",
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "threads_detection": 1,
    }


def max_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
