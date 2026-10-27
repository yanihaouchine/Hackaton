"""Entraînement LightGBM : seuil et hyperparamètres choisis sur la validation uniquement."""

from __future__ import annotations

import time

import numpy as np

from aml.config import SEED, TRAIN_THREADS
from aml.evaluate import best_f1_threshold, pr_auc

# Familles de features (préfixes des noms produits par le moteur C)
FAMILIES = {
    "tx": ("tx_",),
    "kyc": ("kyc_",),
    "profile": ("s_", "r_"),
    "relation": ("p_",),
    "degree": ("g_",),
    "flow": ("f_",),
    "motif": ("m_",),
    "hindsight": ("h_",),  # revue J+1 uniquement
    "propagation": ("z_",),  # étape 2 uniquement (scores de l'étape 1 propagés)
}
ALL_FAMILIES = [f for f in FAMILIES if f not in ("hindsight", "propagation")]  # modèle strict, étape 1

# Ablations cumulatives (baselines B1 → complet), puis retraits d'une famille au modèle complet
ABLATIONS = {
    "B1_intrinseques": ["tx"],
    "B2_profils": ["tx", "profile", "relation"],
    "B3_degres": ["tx", "profile", "relation", "degree"],
    "B4_motifs": ["tx", "profile", "relation", "degree", "motif"],
    "B5_flux": ["tx", "profile", "relation", "degree", "motif", "flow"],
    "B6_complet": ALL_FAMILIES,
    "complet_sans_flux": [f for f in ALL_FAMILIES if f != "flow"],
    "complet_sans_motifs": [f for f in ALL_FAMILIES if f != "motif"],
    "complet_sans_kyc": [f for f in ALL_FAMILIES if f != "kyc"],
}

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": TRAIN_THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def select(names: list[str], families: list[str]) -> list[int]:
    prefixes = tuple(p for f in families for p in FAMILIES[f])
    return [i for i, n in enumerate(names) if n.startswith(prefixes)]


def subsample_rows(y: np.ndarray, lo: int, hi: int, neg_rate: float, seed: int) -> np.ndarray:
    """Toutes les transactions illicites + une fraction des licites (indices absolus, triés)."""
    rng = np.random.default_rng(seed)
    yy = y[lo:hi]
    keep = (yy == 1) | (rng.random(hi - lo) < neg_rate)
    return lo + np.flatnonzero(keep)


def fit(X, y, names, train_rows, val_range, cols, seed=SEED, params=None, rounds=3000, patience=100):
    """Entraîne sur train_rows, arrêt précoce sur la PR-AUC de validation ; renvoie (booster, infos)."""
    import lightgbm as lgb

    p = dict(DEFAULT_PARAMS, **(params or {}), seed=seed)
    t0 = time.perf_counter()
    xtr = np.asarray(X[train_rows][:, cols], dtype=np.float32)
    lo, hi = val_range
    xva = np.asarray(X[lo:hi][:, cols], dtype=np.float32)
    fn = [names[i] for i in cols]
    dtr = lgb.Dataset(xtr, y[train_rows], feature_name=fn, free_raw_data=True)
    dva = lgb.Dataset(xva, y[lo:hi], reference=dtr, free_raw_data=True)
    booster = lgb.train(
        p, dtr, num_boost_round=rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(patience, verbose=False)]
    )
    s_val = booster.predict(xva, num_iteration=booster.best_iteration)
    info = {
        "seed": seed,
        "best_iteration": int(booster.best_iteration),
        "train_rows": int(len(train_rows)),
        "train_seconds": round(time.perf_counter() - t0, 1),
        "val_pr_auc": pr_auc(y[lo:hi], s_val),
        "val_best_f1": best_f1_threshold(y[lo:hi], s_val),
        "params": p,
    }
    return booster, s_val, info
