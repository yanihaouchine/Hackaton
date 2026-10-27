"""Rejeu en flux : le moteur C traite les transactions dans l'ordre, score chacune à l'arrivée et lève des
alertes ; les alertes sont regroupées en cas, typées et transformées en rapports à la demande."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from aml.config import EngineConfig
from aml.data import load
from aml.engine import TX_FIELDS, Engine
from aml.forest import c_raw
from aml.model import load_bundle


@dataclass
class Candidates:
    """Transactions du jour gardées pour la revue J+1 (score d'étape 1 au-dessus d'un seuil bas)."""

    idx: list
    feats: list
    seq: list
    scores: list


@dataclass
class Chunk:
    lo: int
    hi: int
    alerts: np.ndarray  # indices absolus des transactions en alerte
    scores: np.ndarray  # probabilités brutes de ces alertes
    feats: np.ndarray  # features moteur de ces alertes
    seconds: float


class Monitor:
    def __init__(self, variant: str, model: str = "strict", cascade: bool = True):
        from aml.model import load_cascade

        self.s = load(variant)
        self.b = load_bundle(model)
        cfg = EngineConfig(**self.b.info["engine_config"])
        self.eng = Engine.for_stream(self.s, cfg)
        # étape 2 : le moteur score à l'arrivée avec le modèle d'étape 1 (propagation), puis la forêt
        # d'étape 2 donne le score final ; sinon, la forêt du modèle est évaluée directement dans le moteur
        self.stage1 = load_bundle(self.b.info.get("stage1", "strict")) if self.b.info.get("stage2") else self.b
        self.eng.set_forest(self.stage1.forest)
        c = load_cascade(self.stage1) if cascade else None
        if c is not None:
            self.eng.set_cascade(*c)
        self.cascade = c is not None
        self.pos = 0

    def snapshot_path(self, pos: int):
        from aml.config import DATA

        tag = f"{self.b.forest.digest()}_{'c' if self.cascade else 'n'}_{pos}"
        return DATA / "state" / self.s.meta["variant"] / f"{tag}.snap"

    def warm(self, hi: int, progress: Callable[[int, int], None] | None = None, chunk: int = 1 << 18,
             use_snapshot: bool = True) -> float:  # fmt: skip
        """Reconstitue l'état sur l'historique [pos, hi) sans lever d'alerte (mise en route du système).
        Si un instantané de cet état existe, il est restauré ; sinon l'état est calculé puis sauvegardé."""
        t0 = time.perf_counter()
        snap = self.snapshot_path(hi)
        if use_snapshot and self.pos == 0 and snap.exists():
            self.eng.restore(snap)
            self.pos = hi
            self.restored = True
            return time.perf_counter() - t0
        self.restored = False
        buf = np.empty((chunk, self.eng.n_features), dtype=np.float32)
        while self.pos < hi:
            b = min(hi, self.pos + chunk)
            cols = {k: getattr(self.s, k)[self.pos : b] for k, _ in TX_FIELDS}
            self.eng.process(cols, buf[: b - self.pos])
            self.pos = b
            if progress:
                progress(self.pos, hi)
        if use_snapshot:
            snap.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_bytes = self.eng.snapshot(snap)
        return time.perf_counter() - t0

    def replay(self, hi: int, chunk: int = 4096, keep: Candidates | None = None,
               keep_min: float = 0.05) -> Iterator[Chunk]:  # fmt: skip
        """Traite [pos, hi) par micro-lots ; chaque transaction est scorée avant la suivante (C).
        Si `keep` est fourni, les transactions de score >= keep_min y sont conservées (revue J+1)."""
        buf = np.empty((chunk, self.eng.n_features), dtype=np.float32)
        sc = np.empty(chunk, dtype=np.float64)
        seq = np.empty(chunk, dtype=np.int64)
        thr = self.b.threshold
        while self.pos < hi:
            b = min(hi, self.pos + chunk)
            n = b - self.pos
            cols = {k: getattr(self.s, k)[self.pos : b] for k, _ in TX_FIELDS}
            t0 = time.perf_counter()
            self.eng.process(cols, buf[:n], scores=sc[:n], seq_at=seq[:n] if keep is not None else None)
            if self.stage1 is not self.b:  # score final de l'étape 2 (évaluateur C, par lot)
                sc[:n] = 1.0 / (1.0 + np.exp(-c_raw(self.b.forest, buf[:n])))
            dt = time.perf_counter() - t0
            hit = np.flatnonzero(sc[:n] >= thr)
            if keep is not None:
                c = np.flatnonzero(sc[:n] >= keep_min)
                keep.idx.append(self.pos + c)
                keep.feats.append(buf[c].copy())
                keep.seq.append(seq[c].copy())
                keep.scores.append(sc[c].copy())
            yield Chunk(self.pos, b, self.pos + hit, sc[hit].copy(), buf[hit].copy(), dt)
            self.pos = b

    def review(self, keep: Candidates, j1) -> tuple[np.ndarray, np.ndarray]:
        """Revue J+1 à l'instant courant (fin de journée) : features rétrospectives des candidats puis score du
        modèle J+1. Renvoie (indices des transactions, probabilités J+1)."""
        if not keep.idx:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        idx = np.concatenate(keep.idx)
        x = np.concatenate(keep.feats)
        seq = np.concatenate(keep.seq)
        cols = {k: getattr(self.s, k)[idx] for k, _ in TX_FIELDS}
        h = self.eng.hindsight(cols, seq)
        xh = np.ascontiguousarray(np.hstack([x, h]), dtype=np.float32)
        return idx, 1.0 / (1.0 + np.exp(-c_raw(j1.forest, xh)))
