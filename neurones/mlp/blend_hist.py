"""Mélange forêt + modèle séquentiel, réglé sur la VALIDATION seulement (même méthode que mlp.blend).

score = w × log-cote(forêt) + (1 − w) × log-cote(modèle séquentiel), w de 0 à 1 par pas de 0,05, choisi sur la
PR-AUC de validation ; seuil au F1 maximal de validation ; contrôle croisé par jour. Le test n'est pas lu.
Usage (depuis neurones/) : python -m mlp.blend_hist [--model hist_h64-32_g16_s42]  ->  results/val_blend_hist.json
"""

from __future__ import annotations

import argparse

import numpy as np

from mlp.blend import at_threshold, choose
from mlp.common import best_f1_threshold, load_features, load_stream, pr_auc, write_result
from mlp.forest import load_forest
from mlp.hist_scores import logits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="hist_h64-32_g16_s42")
    a = p.parse_args()

    stream, lab = load_stream()
    lo, hi = stream.split_range("val")
    y = lab["y"][lo:hi].astype(bool)
    day = np.asarray(stream.t[lo:hi]) // 1440
    zm = logits(a.model, "val")
    forest_raw, _, forest_name = load_forest()
    X, _, _ = load_features()
    zf = forest_raw(np.asarray(X[lo:hi], dtype=np.float32))

    w, thr, grid = choose(y, zf, zm)
    s = w * zf + (1 - w) * zm

    def alone(z: np.ndarray) -> dict:
        return {"pr_auc": pr_auc(y, z), **at_threshold(y, z, best_f1_threshold(y, z)["threshold"])}

    full = {"w": w, "threshold": thr, "pr_auc": pr_auc(y, s), **at_threshold(y, s, thr),
            "forest_alone": alone(zf), "hist_alone": alone(zm)}  # fmt: skip
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
    write_result("val_blend_hist", {"model": a.model, "forest": forest_name, "split": "val", "grid": grid,
                                    "full_val": full, "cross_day": cross}, seed=42)  # fmt: skip
    for k, d in (("mélange", full), ("forêt seule", full["forest_alone"]), ("séquentiel seul", full["hist_alone"])):
        print(f"{k:16s} PR-AUC {d['pr_auc']:.4f} F1 {d['f1']:.4f} (P {d['precision']:.3f} R {d['recall']:.3f}, "
              f"{d['tp']} VP, {d['fp']} FP)")  # fmt: skip
    print(f"w retenu {w} ; grille :", ", ".join(f"{g['w']}: {g['pr_auc']:.4f}" for g in grid))
    for c in cross:
        b, f = c["blend"], c["forest_alone"]
        print(
            f"croisé jour {c['fit_day']}→{c['eval_day']} (w {c['w']}) : mélange F1 {b['f1']:.4f} PR-AUC "
            f"{b['pr_auc']:.4f} | forêt F1 {f['f1']:.4f} PR-AUC {f['pr_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
