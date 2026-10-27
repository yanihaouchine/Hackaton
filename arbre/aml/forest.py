"""Export LightGBM → forêt aplatie, évaluée par le moteur C (ou numpy), sans dépendance à l'inférence.

Format disque (.amlz) : archive numpy compressée. Les seuils sont stockés en float32 arrondis vers le
bas (le plus grand float32 ≤ seuil double) : pour une entrée float32, x ≤ seuil32 ⇔ x ≤ seuil, la
décision est donc exactement celle de LightGBM.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Forest:
    root: np.ndarray
    feat: np.ndarray  # index de feature du moteur, -1 = feuille
    thr: np.ndarray
    left: np.ndarray
    right: np.ndarray
    default_left: np.ndarray
    value: np.ndarray  # valeur de feuille ; pour un nœud interne, valeur moyenne (internal_value)
    engine_features: list[str]
    meta: dict = field(default_factory=dict)
    _keep: list = field(default_factory=list, repr=False)

    @property
    def n_trees(self) -> int:
        return len(self.root)

    @property
    def n_nodes(self) -> int:
        return len(self.feat)

    @property
    def n_class(self) -> int:
        return int(self.meta.get("num_class", 1))

    def raw_multiclass(self, x: np.ndarray) -> np.ndarray:
        """Scores bruts par classe (arbres entrelacés : arbre k → classe k mod K) ; évaluateur C si disponible."""
        k = self.n_class
        x = np.ascontiguousarray(np.atleast_2d(x), dtype=np.float32)
        out = np.zeros((len(x), k), dtype=np.float64)
        for j in range(k):
            sub = Forest(self.root[j::k].copy(), self.feat, self.thr, self.left, self.right, self.default_left,
                         self.value, self.engine_features)  # fmt: skip
            try:
                out[:, j] = c_raw(sub, x)
            except (OSError, RuntimeError):
                out[:, j] = sub.raw(x)
        return out

    def predict_proba_multiclass(self, x: np.ndarray) -> np.ndarray:
        r = self.raw_multiclass(x)
        r -= r.max(axis=1, keepdims=True)
        e = np.exp(r)
        return e / e.sum(axis=1, keepdims=True)

    def saabas(self, x: np.ndarray) -> np.ndarray:
        """Contributions par feature du moteur (méthode de Saabas, repli sans lightgbm) ; dernière colonne = biais."""
        x = np.asarray(x, dtype=np.float32)
        nf = len(self.engine_features)
        out = np.zeros((len(x), nf + 1), dtype=np.float64)
        for i in range(len(x)):
            for r in self.root:
                n = r
                out[i, nf] += self.value[n]
                while self.feat[n] >= 0:
                    v = x[i, self.feat[n]]
                    go_left = (self.default_left[n] > 0) if np.isnan(v) else (float(v) <= self.thr[n])
                    c = self.left[n] if go_left else self.right[n]
                    out[i, self.feat[n]] += self.value[c] - self.value[n]
                    n = c
        return out

    def raw(self, x: np.ndarray) -> np.ndarray:
        """Score brut en numpy (référence et repli)."""
        x = np.asarray(x, dtype=np.float32)
        n = len(x)
        out = np.zeros(n, dtype=np.float64)
        rows = np.arange(n)
        for r in self.root:
            node = np.full(n, r, dtype=np.int64)
            while True:
                f = self.feat[node]
                inner = f >= 0
                if not inner.any():
                    break
                idx = np.flatnonzero(inner)
                v = x[rows[idx], f[idx]].astype(np.float64)
                nd = node[idx]
                go_left = np.where(np.isnan(v), self.default_left[nd] > 0, v <= self.thr[nd])
                node[idx] = np.where(go_left, self.left[nd], self.right[nd])
            out += self.value[node]
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.raw(x)))

    def to_c(self):
        from aml.engine import aml_forest_struct

        arrs = {
            "root": np.ascontiguousarray(self.root, dtype=np.int32),
            "feat": np.ascontiguousarray(self.feat, dtype=np.int32),
            "thr": np.ascontiguousarray(self.thr, dtype=np.float64),
            "left": np.ascontiguousarray(self.left, dtype=np.int32),
            "right": np.ascontiguousarray(self.right, dtype=np.int32),
            "default_left": np.ascontiguousarray(self.default_left, dtype=np.uint8),
            "value": np.ascontiguousarray(self.value, dtype=np.float64),
        }
        self._keep = [arrs]
        return aml_forest_struct(self.n_trees, self.n_nodes, arrs)

    def save(self, path: Path) -> int:
        header = json.dumps({"engine_features": self.engine_features, "meta": self.meta}).encode()
        thr32 = self.thr.astype(np.float32)
        thr32 = np.where(thr32.astype(np.float64) > self.thr, np.nextafter(thr32, np.float32(-np.inf)), thr32)
        with open(path, "wb") as fh:
            np.savez_compressed(
                fh,
                header=np.frombuffer(header, dtype=np.uint8),
                root=self.root.astype(np.int32),
                feat=self.feat.astype(np.int16),
                thr=thr32.astype(np.float32),
                right=self.right.astype(np.int32),  # gauche = nœud suivant (ordre préfixe)
                default_left=self.default_left.astype(np.uint8),
                value=self.value.astype(np.float64),  # feuilles + nœuds internes (contributions Saabas)
            )
        return path.stat().st_size

    @classmethod
    def load(cls, path: Path) -> Forest:
        with open(path, "rb") as fh:
            z = np.load(fh)
            header = json.loads(z["header"].tobytes().decode())
            feat = z["feat"].astype(np.int32)
            value = z["value"].astype(np.float64)
            n = len(feat)
            return cls(
                root=z["root"].astype(np.int32),
                feat=feat,
                thr=z["thr"].astype(np.float64),
                left=np.arange(1, n + 1, dtype=np.int32),
                right=z["right"].astype(np.int32),
                default_left=z["default_left"].astype(np.uint8),
                value=value,
                engine_features=header["engine_features"],
                meta=header["meta"],
            )

    def digest(self) -> str:
        h = hashlib.sha256()
        for a in (self.root, self.feat, self.thr, self.right, self.value):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()[:12]


def from_lightgbm(booster, cols: list[int], engine_features: list[str], num_iteration: int | None = None,
                  meta: dict | None = None) -> Forest:  # fmt: skip
    """Aplatit les arbres en ordre préfixe (gauche = nœud suivant) ; indices de features → moteur."""
    dump = booster.dump_model(num_iteration=num_iteration)
    objective = dump.get("objective", "binary").split()[0]
    if objective not in ("binary", "multiclass"):
        raise ValueError(f"objectif non pris en charge : {objective}")
    meta = dict(meta or {}, objective=objective, num_class=int(dump.get("num_class", 1)))
    feat, thr, left, right, dl, value, roots = [], [], [], [], [], [], []

    def walk(node: dict) -> int:
        i = len(feat)
        feat.append(-1), thr.append(0.0), left.append(-1), right.append(-1), dl.append(0), value.append(0.0)
        if "leaf_value" in node and "split_feature" not in node:
            value[i] = float(node["leaf_value"])
            return i
        value[i] = float(node.get("internal_value", 0.0))
        if node.get("decision_type", "<=") != "<=":
            raise ValueError("splits catégoriels non pris en charge")
        mt = node.get("missing_type", "None")
        if mt not in ("None", "NaN"):
            raise ValueError(f"missing_type {mt} non pris en charge")
        t = float(node["threshold"])
        feat[i] = cols[int(node["split_feature"])]
        thr[i] = t
        dl[i] = int(bool(node["default_left"])) if mt == "NaN" else int(0.0 <= t)  # None : NaN traité comme 0
        left[i] = walk(node["left_child"])
        right[i] = walk(node["right_child"])
        return i

    for tree in dump["tree_info"]:
        roots.append(walk(tree["tree_structure"]))
    f = Forest(
        root=np.array(roots, dtype=np.int32),
        feat=np.array(feat, dtype=np.int32),
        thr=np.array(thr, dtype=np.float64),
        left=np.array(left, dtype=np.int32),
        right=np.array(right, dtype=np.int32),
        default_left=np.array(dl, dtype=np.uint8),
        value=np.array(value, dtype=np.float64),
        engine_features=list(engine_features),
        meta=meta,
    )
    assert np.all((f.feat < 0) | (f.left == np.arange(f.n_nodes) + 1)), "ordre préfixe attendu"
    return f


def c_raw(forest: Forest, x: np.ndarray) -> np.ndarray:
    """Score brut par l'évaluateur C (même code que le moteur en ligne)."""
    from aml.engine import load_lib

    lib = load_lib()
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty(len(x), dtype=np.float64)
    st = forest.to_c()
    lib.aml_forest_raw(ctypes.byref(st), x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x), x.shape[1],
                       out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))  # fmt: skip
    return out
