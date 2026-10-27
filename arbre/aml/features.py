"""Matrice de features du flux complet, calculée par le moteur C (même code qu'en inférence)."""

from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aml.config import DATA_FEAT, EngineConfig
from aml.data import load
from aml.engine import Engine


def feature_digest(cfg: EngineConfig) -> str:
    """Empreinte des features : paramètres du moteur + liste des features qu'il produit."""
    import hashlib

    from aml.engine import feature_names

    return hashlib.sha256((cfg.digest() + "|" + ",".join(feature_names())).encode()).hexdigest()[:12]


def feature_dir(variant: str, cfg: EngineConfig) -> Path:
    return DATA_FEAT / variant / feature_digest(cfg)


def compute(variant: str, cfg: EngineConfig | None = None, progress: bool = True) -> Path:
    cfg = cfg or EngineConfig()
    s = load(variant)
    out_dir = feature_dir(variant, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    eng = Engine.for_stream(s, cfg)
    X = np.lib.format.open_memmap(out_dir / "X.npy", mode="w+", dtype=np.float32, shape=(s.n, eng.n_features))
    t0 = time.perf_counter()
    eng.process_stream(s, 0, s.n, out=X, chunk=1 << 20, progress=progress)
    seconds = time.perf_counter() - t0
    X.flush()
    meta = {
        "variant": variant,
        "config": asdict(cfg),
        "config_digest": feature_digest(cfg),
        "engine_config_digest": cfg.digest(),
        "feature_names": eng.names,
        "n_tx": s.n,
        "seconds": round(seconds, 3),
        "tx_per_s": round(s.n / seconds),
        "state_bytes": eng.state_bytes(),
        "stats": eng.stats(),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    eng.close()
    return out_dir


def load_features(variant: str, cfg: EngineConfig | None = None) -> tuple[np.ndarray, list[str], dict]:
    d = feature_dir(variant, cfg or EngineConfig())
    meta = json.loads((d / "meta.json").read_text())
    return np.load(d / "X.npy", mmap_mode="r"), meta["feature_names"], meta


def compute_hindsight(variant: str, cfg: EngineConfig | None = None, progress: bool = True) -> Path:
    """Revue J+1 : jour par jour, features rétrospectives calculées à minuit (fin du jour de l'opération)."""
    from aml.engine import TX_FIELDS, hindsight_names

    cfg = cfg or EngineConfig()
    s = load(variant)
    out_dir = feature_dir(variant, cfg)
    eng = Engine.for_stream(s, cfg)
    names = hindsight_names()
    H = np.lib.format.open_memmap(out_dir / "H.npy", mode="w+", dtype=np.float32, shape=(s.n, len(names)))
    t0 = time.perf_counter()
    days = np.asarray(s.t) // 1440
    bounds = np.r_[0, np.flatnonzero(np.diff(days)) + 1, s.n]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        cols = {k: getattr(s, k)[lo:hi] for k, _ in TX_FIELDS}
        seq = np.empty(hi - lo, dtype=np.int64)
        eng.process(cols, seq_at=seq)  # état jusqu'à la fin du jour = instant de décision
        H[lo:hi] = eng.hindsight(cols, seq)
        if progress:
            print(f"  jour {int(days[lo])}: {hi - lo:,d} tx ({time.perf_counter() - t0:.1f} s)", flush=True)
    H.flush()
    meta = {"hindsight_names": names, "decision": "fin du jour de l'opération (minuit)",
            "seconds": round(time.perf_counter() - t0, 3), "config_digest": feature_digest(cfg)}  # fmt: skip
    (out_dir / "hindsight.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return out_dir


def load_hindsight(variant: str, cfg: EngineConfig | None = None) -> tuple[np.ndarray, list[str]]:
    d = feature_dir(variant, cfg or EngineConfig())
    meta = json.loads((d / "hindsight.json").read_text())
    return np.load(d / "H.npy", mmap_mode="r"), meta["hindsight_names"]


def compute_stage2(variant: str, m1: str = "strict", m1a: str = "strict_a", m1b: str = "strict_b",
                   cfg: EngineConfig | None = None) -> Path:  # fmt: skip
    """Features avec propagation (z_*) : le moteur score chaque transaction à l'arrivée avec un modèle d'étape 1
    et propage ces scores aux transactions suivantes. Sur le train, validation croisée temporelle (jours 0–2
    scorés par le modèle appris sur les jours 3–5, et inversement) ; ensuite, modèle d'étape 1 complet."""
    from aml.model import load_bundle

    cfg = cfg or EngineConfig()
    s = load(variant)
    out_dir = feature_dir(variant, cfg)
    eng = Engine.for_stream(s, cfg)
    X = np.lib.format.open_memmap(out_dir / "X_stage2.npy", mode="w+", dtype=np.float32,
                                  shape=(s.n, eng.n_features))  # fmt: skip
    i3 = int(np.searchsorted(s.t, 3 * 1440))
    i6, _ = s.split_range("val")
    plan = [(0, i3, m1b), (i3, i6, m1a), (i6, s.n, m1)]
    t0 = time.perf_counter()
    for lo, hi, name in plan:
        eng.set_forest(load_bundle(name).forest)
        eng.process_stream(s, lo, hi, out=X[lo:hi], chunk=1 << 20)
    X.flush()
    meta = {"plan": [[lo, hi, n] for lo, hi, n in plan], "seconds": round(time.perf_counter() - t0, 3),
            "feature_names": eng.names, "config_digest": feature_digest(cfg)}  # fmt: skip
    (out_dir / "stage2.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out_dir
