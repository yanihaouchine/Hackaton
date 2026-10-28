"""Complémentarité du MLP livré et de la forêt strict de arbre/, sur la VALIDATION uniquement (le test n'est pas lu).

Chaque modèle est pris à son seuil de validation. On compte les fraudes et les tentatives attrapées par l'un,
l'autre, les deux ou aucun, le recouvrement des fausses alertes, et la corrélation des rangs des scores. Les bornes
« union » et « intersection » disent ce qu'une combinaison pourrait au mieux gagner en rappel ou en précision.
Usage (depuis neurones/) : python -m mlp.complementarity  ->  results/val_complementarity.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort

from mlp.common import ARTIFACTS, load_features, load_stream, pr_auc, write_result


def ranks(s: np.ndarray) -> np.ndarray:
    r = np.empty(len(s))
    r[np.argsort(s, kind="stable")] = np.arange(len(s))
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def prf(alert: np.ndarray, y: np.ndarray) -> dict:
    tp, fp, fn = int(np.sum(alert & y)), int(np.sum(alert & ~y)), int(np.sum(~alert & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def main() -> None:
    from aml.forest import c_raw
    from aml.model import load_bundle

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    a = p.parse_args()

    X, _, cols = load_features()
    stream, lab = load_stream()
    lo, hi = stream.split_range("val")
    x_all = np.asarray(X[lo:hi], dtype=np.float32)
    x = np.ascontiguousarray(x_all[:, cols])
    y = lab["y"][lo:hi].astype(bool)
    att = lab["attempt"][lo:hi]

    meta = json.loads((ARTIFACTS / a.model / "meta.json").read_text())
    sess = ort.InferenceSession(str(ARTIFACTS / a.model / "model.onnx"), providers=["CPUExecutionProvider"])
    s_mlp = np.concatenate([sess.run(None, {"features": x[i : i + 65536]})[0] for i in range(0, len(x), 65536)])
    bundle = load_bundle("strict")
    s_for = 1.0 / (1.0 + np.exp(-c_raw(bundle.forest, x_all)))
    a_mlp, a_for = s_mlp >= meta["threshold"], s_for >= bundle.threshold

    # Fraudes (transactions illicites) : qui les attrape ?
    tx = {"both": int(np.sum(y & a_mlp & a_for)), "only_mlp": int(np.sum(y & a_mlp & ~a_for)),
          "only_forest": int(np.sum(y & ~a_mlp & a_for)), "neither": int(np.sum(y & ~a_mlp & ~a_for)),
          "n_illicit": int(y.sum())}  # fmt: skip
    # Fausses alertes : les mêmes ?
    fp = {"both": int(np.sum(~y & a_mlp & a_for)), "only_mlp": int(np.sum(~y & a_mlp & ~a_for)),
          "only_forest": int(np.sum(~y & ~a_mlp & a_for))}  # fmt: skip
    # Tentatives (au moins une transaction en alerte dans la période)
    ids = np.unique(att[att >= 0])
    det_m = {int(i) for i in ids if a_mlp[att == i].any()}
    det_f = {int(i) for i in ids if a_for[att == i].any()}
    attempts = {"n": int(len(ids)), "both": len(det_m & det_f), "only_mlp": len(det_m - det_f),
                "only_forest": len(det_f - det_m), "neither": int(len(ids)) - len(det_m | det_f)}  # fmt: skip

    out = {
        "model": a.model,
        "split": "val",
        "thresholds": {"mlp": meta["threshold"], "forest": bundle.threshold},
        "mlp": {**prf(a_mlp, y), "pr_auc": pr_auc(y, s_mlp)},
        "forest": {**prf(a_for, y), "pr_auc": pr_auc(y, s_for)},
        "illicit_tx": tx,
        "false_alerts": fp,
        "attempts": attempts,
        "bounds": {"union_or": prf(a_mlp | a_for, y), "intersection_and": prf(a_mlp & a_for, y)},
        "spearman_scores": {
            "all": spearman(s_mlp, s_for),
            "illicit": spearman(s_mlp[y], s_for[y]),
            "licit_alerted_by_either": spearman(s_mlp[~y & (a_mlp | a_for)], s_for[~y & (a_mlp | a_for)]),
        },  # fmt: skip
    }
    write_result("val_complementarity", out, seed=42)
    print(json.dumps({k: v for k, v in out.items() if k not in ("model", "split")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
