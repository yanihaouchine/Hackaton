"""Log-cotes du modèle séquentiel, relues depuis artifacts/<nom>/logit_<période>.npy (écrites par
`python -m mlp.train_hist --dump <nom>`), sans importer torch : sur macOS, torch et lightgbm embarquent chacun
leur libomp et ne peuvent pas être chargés dans le même processus."""

from __future__ import annotations

import json

import numpy as np

from mlp.common import ARTIFACTS


def logits(name: str, split: str) -> np.ndarray:
    path = ARTIFACTS / name / f"logit_{split}.npy"
    if not path.exists():
        raise SystemExit(f"{path} manquant : lancer d'abord python -m mlp.train_hist --dump {name}")
    return np.load(path)


def threshold(name: str) -> float:
    """Seuil de validation du modèle, en probabilité."""
    return json.loads((ARTIFACTS / name / "meta.json").read_text())["threshold"]
