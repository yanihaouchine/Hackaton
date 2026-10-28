"""Option A : mélange des scores du MLP livré et de la forêt strict de arbre/, réglé sur la VALIDATION seulement.

Méthode fixée avant de voir les résultats :
- score = w × log-cote(forêt) + (1 − w) × log-cote(MLP), w de 0 à 1 par pas de 0,05 ;
- w choisi sur la PR-AUC de validation ; seuil au F1 maximal de validation (comme pour les modèles seuls) ;
- contrôle croisé par jour : w et seuil choisis sur le 07/09, mesurés sur le 08/09, et l'inverse, face à la
  forêt seule (seuil choisi de la même façon) sur le même jour.
Le test n'est pas lu. Usage (depuis neurones/) : python -m mlp.blend  ->  results/val_blend.json
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort

from mlp.common import ARTIFACTS, best_f1_threshold, load_features, load_stream, pr_auc, write_result
from mlp.forest import load_forest

WEIGHTS = [round(0.05 * i, 2) for i in range(21)]  # 0 = MLP seul, 1 = forêt seule


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    a = s >= thr
    tp, fp, fn = int(np.sum(a & y)), int(np.sum(a & ~y)), int(np.sum(~a & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def choose(y: np.ndarray, zf: np.ndarray, zm: np.ndarray) -> tuple[float, float, list[dict]]:
    grid = [{"w": w, "pr_auc": pr_auc(y, w * zf + (1 - w) * zm)} for w in WEIGHTS]
    w = max(grid, key=lambda g: g["pr_auc"])["w"]
    return w, best_f1_threshold(y, w * zf + (1 - w) * zm)["threshold"], grid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    a = p.parse_args()

    X, _, cols = load_features()
    stream, lab = load_stream()
    lo, hi = stream.split_range("val")
    x_all = np.asarray(X[lo:hi], dtype=np.float32)
    x = np.ascontiguousarray(x_all[:, cols])
    y = lab["y"][lo:hi].astype(bool)
    day = np.asarray(stream.t[lo:hi]) // 1440

    sess = ort.InferenceSession(str(ARTIFACTS / a.model / "model.onnx"), providers=["CPUExecutionProvider"])
    zm = logit(np.concatenate([sess.run(None, {"features": x[i : i + 65536]})[0] for i in range(0, len(x), 65536)]))
    forest_raw, _, forest_name = load_forest()
    zf = forest_raw(x_all)  # log-cote brute de la forêt

    w, thr, grid = choose(y, zf, zm)
    s = w * zf + (1 - w) * zm
    forest_thr = best_f1_threshold(y, zf)["threshold"]
    full = {"w": w, "threshold": thr, "pr_auc": pr_auc(y, s), **at_threshold(y, s, thr),
            "forest_alone": {"pr_auc": pr_auc(y, zf), **at_threshold(y, zf, forest_thr)}}  # fmt: skip

    cross = []
    days = np.unique(day)
    for d_fit, d_eval in ((days[0], days[1]), (days[1], days[0])):
        f, e = day == d_fit, day == d_eval
        wd, td, _ = choose(y[f], zf[f], zm[f])
        tf = best_f1_threshold(y[f], zf[f])["threshold"]
        cross.append({"fit_day": int(d_fit), "eval_day": int(d_eval), "w": wd,
                      "blend": {"pr_auc": pr_auc(y[e], wd * zf[e] + (1 - wd) * zm[e]),
                                **at_threshold(y[e], wd * zf[e] + (1 - wd) * zm[e], td)},
                      "forest_alone": {"pr_auc": pr_auc(y[e], zf[e]), **at_threshold(y[e], zf[e], tf)}})  # fmt: skip

    method = __doc__.split("Méthode fixée")[1].split("Le test")[0].strip()
    out = {"model": a.model, "forest": forest_name, "split": "val", "method": method,
           "grid": grid, "full_val": full, "cross_day": cross}  # fmt: skip
    write_result("val_blend", out, seed=42)
    print(f"w retenu {w} ; validation : PR-AUC {full['pr_auc']:.4f} F1 {full['f1']:.4f} "
          f"(P {full['precision']:.3f} R {full['recall']:.3f}, {full['tp']} VP, {full['fp']} FP)")  # fmt: skip
    fa = full["forest_alone"]
    print(f"forêt seule           : PR-AUC {fa['pr_auc']:.4f} F1 {fa['f1']:.4f} "
          f"(P {fa['precision']:.3f} R {fa['recall']:.3f}, {fa['tp']} VP, {fa['fp']} FP)")  # fmt: skip
    print("grille (w : PR-AUC) :", ", ".join(f"{g['w']}: {g['pr_auc']:.4f}" for g in grid))
    for c in cross:
        b, f = c["blend"], c["forest_alone"]
        print(
            f"croisé jour {c['fit_day']}→{c['eval_day']} (w {c['w']}) : mélange F1 {b['f1']:.4f} PR-AUC "
            f"{b['pr_auc']:.4f} ({b['fp']} FP) | forêt F1 {f['f1']:.4f} PR-AUC {f['pr_auc']:.4f} ({f['fp']} FP)"
        )


if __name__ == "__main__":
    main()
