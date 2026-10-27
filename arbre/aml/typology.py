"""Typologie nommée d'une alerte : classifieur multiclasse (8 typologies + « hors motif ») entraîné sur
les illicites du train à partir des étiquettes de Patterns.txt (entraînement et évaluation uniquement)."""

from __future__ import annotations

import json

import numpy as np

from aml.config import TRAIN_THREADS, TYPOLOGIES
from aml.forest import Forest, from_lightgbm
from aml.model import model_dir
from aml.provenance import write_result

CLASSES = TYPOLOGIES + ["OTHER"]


def _labels(typ: np.ndarray) -> np.ndarray:
    return np.where(typ >= 0, typ, len(TYPOLOGIES)).astype(np.int32)


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> tuple[float, list[float]]:
    f1s = []
    for c in range(k):
        tp = np.sum((p == c) & (y == c))
        fp = np.sum((p == c) & (y != c))
        fn = np.sum((p != c) & (y == c))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan"))
    return float(np.nanmean(f1s)), [float(v) for v in f1s]


def train(variant: str, seed: int = 42) -> dict:
    import lightgbm as lgb

    from aml.data import load, load_labels
    from aml.features import load_features

    s = load(variant)
    lab = load_labels(variant)
    y, typ = lab["y"], lab["typ"]
    X, names, fmeta = load_features(variant)
    tr, va = s.split_range("train"), s.split_range("val")
    rtr = tr[0] + np.flatnonzero(y[tr[0] : tr[1]] == 1)
    rva = va[0] + np.flatnonzero(y[va[0] : va[1]] == 1)
    k = len(CLASSES)
    params = {"objective": "multiclass", "num_class": k, "learning_rate": 0.05, "num_leaves": 15,
              "min_data_in_leaf": 10, "feature_fraction": 0.8, "lambda_l2": 1.0, "num_threads": TRAIN_THREADS,
              "deterministic": True, "force_row_wise": True, "verbose": -1, "seed": seed}  # fmt: skip
    xtr, xva = np.asarray(X[rtr]), np.asarray(X[rva])
    ytr, yva = _labels(typ[rtr]), _labels(typ[rva])
    dtr = lgb.Dataset(xtr, ytr, feature_name=names)
    dva = lgb.Dataset(xva, yva, reference=dtr)
    b = lgb.train(params, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    forest = from_lightgbm(b, list(range(len(names))), names, b.best_iteration,
                           meta={"name": "typology", "classes": CLASSES})  # fmt: skip
    proba = forest.predict_proba_multiclass(xva)
    ref = b.predict(xva, num_iteration=b.best_iteration)
    if np.max(np.abs(proba - ref)) > 1e-6:
        raise RuntimeError("écart forêt multiclasse / LightGBM")
    pred = proba.argmax(1)
    mf1, per = macro_f1(yva, pred, k)
    conf = np.zeros((k, k), dtype=int)
    np.add.at(conf, (yva, pred), 1)
    d = model_dir("typology")
    d.mkdir(parents=True, exist_ok=True)
    size = forest.save(d / "forest.amlz")
    info = {"classes": CLASSES, "best_iteration": int(b.best_iteration), "n_train": int(len(rtr)),
            "n_val": int(len(rva)), "val_accuracy": float((pred == yva).mean()), "val_macro_f1": mf1,
            "val_f1_by_class": dict(zip(CLASSES, per, strict=True)), "forest_bytes": size,
            "engine_features_digest": fmeta["config_digest"]}  # fmt: skip
    (d / "model.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    write_result("val_typology", {**info, "confusion_val": conf.tolist()})
    return info


class TypologyModel:
    def __init__(self):
        d = model_dir("typology")
        self.forest = Forest.load(d / "forest.amlz")
        self.info = json.loads((d / "model.json").read_text())
        self.classes = self.info["classes"]

    def predict(self, x: np.ndarray) -> tuple[list[str], np.ndarray]:
        p = self.forest.predict_proba_multiclass(np.atleast_2d(x))
        return [self.classes[i] for i in p.argmax(1)], p


def evaluate_split(variant: str, split: str = "test", detector: str = "strict") -> dict:
    """Évaluation unique du classifieur figé sur une période : toutes les illicites, puis seulement celles
    que le détecteur a signalées (ce que voit l'analyste)."""
    from aml.data import load, load_labels
    from aml.features import load_features
    from aml.forest import c_raw
    from aml.model import load_bundle

    s = load(variant)
    lab = load_labels(variant)
    X, names, _ = load_features(variant)
    lo, hi = s.split_range(split)
    y, typ = lab["y"][lo:hi], lab["typ"][lo:hi]
    pos = np.flatnonzero(y == 1)
    x = np.ascontiguousarray(X[lo:hi][pos], dtype=np.float32)
    m = TypologyModel()
    _, proba = m.predict(x)
    pred, true = proba.argmax(1), _labels(typ[pos])
    b = load_bundle(detector)
    det = 1.0 / (1.0 + np.exp(-c_raw(b.forest, x))) >= b.threshold
    k = len(CLASSES)
    res = {"split": split, "detector": detector}
    for name, mask in (("toutes_illicites", np.ones(len(pos), bool)), ("illicites_detectees", det)):
        mf1, per = macro_f1(true[mask], pred[mask], k)
        res[name] = {"n": int(mask.sum()), "accuracy": float((pred[mask] == true[mask]).mean()), "macro_f1": mf1,
                     "f1_by_class": dict(zip(CLASSES, per, strict=True))}  # fmt: skip
    # niveau cas : typologie du cas = moyenne des probabilités de ses alertes pondérée par leur score ;
    # vérité = typologie majoritaire parmi ses opérations illicites
    from aml.cases import build_cases

    xs = np.ascontiguousarray(X[lo:hi], dtype=np.float32)
    p_all = 1.0 / (1.0 + np.exp(-c_raw(b.forest, xs)))
    al = np.flatnonzero(p_all >= b.threshold)
    ia = lo + al
    cases = build_cases(ia, np.asarray(s.src[ia]), np.asarray(s.dst[ia]), np.asarray(s.t[ia]),
                        np.asarray(s.usd_paid[ia], dtype=np.float64), b.calibrate(p_all[al]))  # fmt: skip
    _, pa = m.predict(xs[al])
    row = {int(i): r for r, i in enumerate(ia)}
    ok, n = 0, 0
    for c in cases:
        rows = [row[a] for a in c.alerts]
        ill = [a for a in c.alerts if lab["y"][a] == 1]
        if not ill:
            continue
        w = p_all[al][rows][:, None]
        guess = int(((pa[rows] * w).sum(0)).argmax())
        truth = np.bincount(_labels(lab["typ"][ill]), minlength=k).argmax()
        ok += int(guess == truth)
        n += 1
    res["cas_avec_illicite"] = {"n": n, "accuracy": ok / n if n else None}
    write_result(f"eval_typology_{split}", res, unique=True)
    return res
