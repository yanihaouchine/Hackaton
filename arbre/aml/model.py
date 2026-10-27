"""Artefact livré : forêt aplatie + seuil (choisi sur la validation) + quantiles du train pour les raisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aml.config import ARTIFACTS
from aml.forest import Forest

QUANTILES = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def model_dir(name: str) -> Path:
    return ARTIFACTS / "models" / name


@dataclass
class Bundle:
    name: str
    forest: Forest
    info: dict  # seuil, colonnes, taux de sous-échantillonnage, métriques de validation…
    quantiles: dict[str, list[float]]
    path: Path

    @property
    def threshold(self) -> float:
        return float(self.info["threshold"])

    @property
    def cols(self) -> list[int]:
        return list(self.info["cols"])

    def calibrate(self, p: np.ndarray) -> np.ndarray:
        """Corrige le sous-échantillonnage des licites (taux r) : cote réelle = cote du modèle × r."""
        r = float(self.info.get("neg_rate", 1.0))
        p = np.asarray(p, dtype=np.float64)
        return p * r / (p * r + (1.0 - p))

    def percentile(self, feature: str, value: float) -> float:
        """Part approximative (0–1) des transactions d'apprentissage ayant une valeur inférieure."""
        q = self.quantiles.get(feature)
        if not q:
            return float("nan")
        return float(np.interp(value, q, QUANTILES, left=0.0, right=1.0))

    def lgbm_path(self) -> Path:
        return self.path / "lgbm.txt"


def save_bundle(name: str, forest: Forest, info: dict, quantiles: dict, booster=None) -> Path:
    d = model_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    size = forest.save(d / "forest.amlz")
    info = dict(info, forest_bytes=size, forest_digest=forest.digest(), n_trees=forest.n_trees,
                n_nodes=forest.n_nodes)  # fmt: skip
    (d / "model.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    (d / "quantiles.json").write_text(json.dumps({"q": QUANTILES, "features": quantiles}) + "\n")
    if booster is not None:
        booster.save_model(str(d / "lgbm.txt"), num_iteration=info.get("best_iteration"))
    return d


def load_bundle(name: str) -> Bundle:
    d = model_dir(name)
    info = json.loads((d / "model.json").read_text())
    q = json.loads((d / "quantiles.json").read_text())["features"]
    return Bundle(name=name, forest=Forest.load(d / "forest.amlz"), info=info, quantiles=q, path=d)


def train_quantiles(X, names: list[str], rows: np.ndarray, max_rows: int = 400_000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(rows, size=min(max_rows, len(rows)), replace=False))  # échantillon du train
    sample = np.asarray(X[pick], dtype=np.float32)
    qs = np.quantile(sample, QUANTILES, axis=0)
    return {n: [float(v) for v in qs[:, j]] for j, n in enumerate(names)}


def load_cascade(bundle: Bundle) -> tuple[Forest, float] | None:
    """Mini-modèle de pré-filtrage associé au modèle (s'il a été entraîné) et son seuil brut."""
    p = bundle.path / "cascade.amlz"
    if not p.exists():
        return None
    info = json.loads((bundle.path / "cascade.json").read_text())
    return Forest.load(p), float(info["cut_raw"])
