"""Données d'entraînement du MLP, identiques à celles de la forêt de `arbre/`.

On réutilise le code de `arbre/` (lecture seule) pour le flux, les étiquettes, les features du moteur C,
le découpage temporel, le choix des colonnes du modèle « strict » et le sous-échantillonnage des licites.
Ce module n'ajoute que la normalisation, dont les statistiques sont calculées sur le train uniquement.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # neurones/
REPO = ROOT.parent
ARBRE = REPO / "arbre"
RESULTS = ROOT / "results"
SEED = 42
NEG_RATE = 0.25  # comme le modèle strict de arbre/ : toutes les illicites + 25 % des licites
SPLITS = ("train", "val", "test", "tail")

if str(ARBRE) not in sys.path:
    sys.path.insert(0, str(ARBRE))

# Codes entiers sans ordre (moyen de paiement, devise, heure, pays…) : encodés en one-hot.
# Les catégories binaires (0/1) restent des nombres.
ONE_HOT = ("tx_fmt", "tx_cur", "tx_hour", "tx_round", "kyc_s_etype", "kyc_r_etype", "kyc_s_country", "kyc_r_country")
CLIP = 10.0  # borne des valeurs centrées-réduites (valeurs extrêmes du test)


@dataclass
class Dataset:
    X: np.ndarray  # features du moteur C, (n, 104), mmap
    y: np.ndarray  # étiquettes : entraînement et évaluation seulement
    names: list[str]  # noms des 104 colonnes de X
    cols: list[int]  # colonnes du modèle strict (96)
    ranges: dict[str, tuple[int, int]]  # indices [début, fin) de chaque période dans le flux trié
    features_digest: str


def load(variant: str = "HI-Small") -> Dataset:
    from aml.data import load as load_stream
    from aml.data import load_labels
    from aml.features import load_features
    from aml.train import ALL_FAMILIES, select

    s = load_stream(variant)
    X, names, fmeta = load_features(variant)
    return Dataset(
        X=X,
        y=load_labels(variant)["y"],
        names=names,
        cols=select(names, ALL_FAMILIES),
        ranges={k: s.split_range(k) for k in SPLITS},
        features_digest=fmeta["config_digest"],
    )


def train_rows(ds: Dataset, seed: int = SEED, neg_rate: float = NEG_RATE) -> np.ndarray:
    """Lignes d'entraînement : même tirage que arbre/ pour une même seed."""
    from aml.train import subsample_rows

    return subsample_rows(ds.y, *ds.ranges["train"], neg_rate=neg_rate, seed=seed)


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Écrase les grandes valeurs (comptages, durées) en gardant le signe : -1 (« jamais ») reste distinct de 0."""
    return np.sign(x) * np.log1p(np.abs(x))


@dataclass
class Preprocessing:
    """Passage des features brutes (colonnes du modèle) aux entrées du réseau. Paramètres appris sur le train."""

    names: list[str]  # noms des colonnes d'entrée, dans l'ordre
    num_idx: list[int]  # positions (parmi `names`) des features numériques
    num_mean: list[float]
    num_std: list[float]
    cat_idx: list[int]  # positions des features one-hot
    cat_sizes: list[int]  # nombre de catégories vues sur le train (code inconnu → vecteur nul)

    @property
    def n_inputs(self) -> int:
        return len(self.num_idx) + sum(self.cat_sizes)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x : (n, len(names)) en float32 → (n, n_inputs) en float32."""
        x = np.asarray(x, dtype=np.float32)
        z = (signed_log1p(x[:, self.num_idx]) - np.float32(self.num_mean)) / np.float32(self.num_std)
        parts = [np.clip(z, -CLIP, CLIP)]
        for j, k in zip(self.cat_idx, self.cat_sizes, strict=True):
            parts.append((x[:, j : j + 1] == np.arange(k, dtype=np.float32)).astype(np.float32))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Preprocessing:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def fit_preprocessing(x_train: np.ndarray, names: list[str]) -> Preprocessing:
    """x_train : lignes d'entraînement uniquement (aucune statistique sur la validation ou le test)."""
    x = np.asarray(x_train, dtype=np.float32)
    cat_idx = [i for i, n in enumerate(names) if n in ONE_HOT]
    num_idx = [i for i in range(len(names)) if i not in cat_idx]
    f = signed_log1p(x[:, num_idx]).astype(np.float64)
    std = f.std(axis=0)
    return Preprocessing(
        names=list(names),
        num_idx=num_idx,
        num_mean=f.mean(axis=0).tolist(),
        num_std=np.where(std > 1e-6, std, 1.0).tolist(),  # colonne constante : on ne divise pas par 0
        cat_idx=cat_idx,
        cat_sizes=[int(x[:, i].max()) + 1 for i in cat_idx],
    )


def rows_of(ds: Dataset, rows: np.ndarray | tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Features brutes (colonnes du modèle) et étiquettes, pour des indices ou une plage [début, fin)."""
    if isinstance(rows, tuple):
        lo, hi = rows
        return np.asarray(ds.X[lo:hi])[:, ds.cols], np.asarray(ds.y[lo:hi])
    return np.asarray(ds.X[rows])[:, ds.cols], np.asarray(ds.y[rows])
