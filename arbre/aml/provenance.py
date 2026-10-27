"""Traçabilité des chiffres : chaque résultat écrit dans results/ porte date, commit, seed et machine."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aml.config import RESULTS, ROOT, SEED


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "inconnu"


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "inconnu"


def run_info(seed: int = SEED) -> dict:
    versions = {}
    for mod in ("numpy", "polars", "lightgbm"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            pass
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
        "seed": seed,
        "machine": {
            "hostname": platform.node(),
            "arch": platform.machine(),
            "cpu": _cpu_model(),
            "n_cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "versions": versions,
    }


def write_result(name: str, payload: dict, seed: int = SEED, unique: bool = False) -> Path:
    """Écrit results/<name>.json ; unique=True ajoute un horodatage pour ne jamais écraser (test)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    if unique:
        name = f"{name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    path = RESULTS / f"{name}.json"
    doc = {"run": run_info(seed), **payload}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=_jsonable) + "\n")
    return path


def _jsonable(o):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"non sérialisable : {type(o)}")
