"""Chemins, accès en lecture seule au code de arbre/ et chargement des features épinglées."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # neurones/
REPO = ROOT.parent  # racine du dépôt : data/ partagé
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

# Code de arbre/ utilisé en lecture seule (données préparées, sous-échantillonnage, métriques).
sys.path.insert(0, str(REPO / "arbre"))
from aml.data import load, load_labels  # noqa: E402
from aml.evaluate import best_f1_threshold, evaluate, pr_auc  # noqa: E402, F401
from aml.provenance import run_info  # noqa: E402
from aml.train import subsample_rows  # noqa: E402, F401

VARIANT = "HI-Small"
# Empreinte des features du moteur de arbre/ (config + noms) : celle de leur modèle livré.
FEATURE_DIGEST = "03a26672a4ef"
EXCLUDED_PREFIXES = ("z_", "h_")  # z_ : scores de la forêt ; h_ : informations postérieures


def load_features(variant: str = VARIANT) -> tuple[np.ndarray, list[str], list[int]]:
    """Matrice brute en mmap, noms de toutes les colonnes, indices des colonnes utilisées par le MLP."""
    d = REPO / "data" / "features" / variant / FEATURE_DIGEST
    meta = json.loads((d / "meta.json").read_text())
    if meta["config_digest"] != FEATURE_DIGEST:
        raise RuntimeError(f"empreinte inattendue : {meta['config_digest']}")
    names = meta["feature_names"]
    cols = [i for i, n in enumerate(names) if not n.startswith(EXCLUDED_PREFIXES)]
    return np.load(d / "X.npy", mmap_mode="r"), names, cols


def load_stream(variant: str = VARIANT):
    """Flux préparé (temps, découpages) et étiquettes : y, attempt, typ."""
    return load(variant), load_labels(variant)


def provenance(seed: int) -> dict:
    info = run_info(seed)
    import torch

    info["versions"]["torch"] = torch.__version__
    info["machine"]["platform"] = platform.platform()
    info["feature_digest"] = FEATURE_DIGEST
    return info


def write_result(name: str, payload: dict, seed: int) -> Path:
    """Écrit results/<name>.json avec la provenance (date, commit, seed, machine)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    doc = {"run": provenance(seed), "written": datetime.now().isoformat(timespec="seconds"), **payload}
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=_jsonable) + "\n")
    return path


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
