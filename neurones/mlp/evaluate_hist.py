"""Évaluation figée du modèle séquentiel (seuil de validation) et du mélange forêt + séquentiel (poids et seuil de
results/val_blend_hist.json) sur test, traîne et protocole littérature. Même protocole que mlp.evaluate.

Le test n'est lu qu'ici, une fois par configuration figée ; chaque lecture écrit un fichier horodaté.
Usage (depuis neurones/) : python -m mlp.evaluate_hist --models hist_h64-32_g16_s42 [...] [--blend]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from mlp.common import RESULTS, evaluate, load_features, load_stream, write_result
from mlp.evaluate import KS, SPLITS, attempt_starts
from mlp.forest import load_forest
from mlp.hist_scores import logits, threshold


def assess(name: str, scores: dict, thr: float, stream, lab, starts) -> dict:
    from aml.cases import build_cases
    from aml.evaluate import case_report

    y, t_all, out = lab["y"], np.asarray(stream.t), {}
    for sp, s in scores.items():
        lo, hi = stream.split_range(sp)
        t = t_all[lo:hi]
        res = evaluate(y[lo:hi], s, thr, t, lab["attempt"][lo:hi], lab["typ"][lo:hi], ks=KS, attempt_start=starts)
        al = np.flatnonzero(s >= thr)
        ia = lo + al
        prio = 1.0 / (1.0 + np.exp(-s[al]))
        cases = build_cases(ia, np.asarray(stream.src[ia]), np.asarray(stream.dst[ia]), t[al],
                            np.asarray(stream.usd_paid[ia], dtype=np.float64), prio)  # fmt: skip
        res["cases"] = case_report(cases, lambda idx: y[np.asarray(idx)] == 1)
        out[sp] = res
        print(f"{name} [{sp:8s}] P {res['precision']:.4f} R {res['recall']:.4f} F1 {res['f1']:.4f} "
              f"IC95 {res['f1_ci95'][0]:.3f}-{res['f1_ci95'][1]:.3f} PR-AUC {res['pr_auc']:.4f} "
              f"alertes {res['n_alerts']}", flush=True)  # fmt: skip
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", default=["hist_h64-32_g16_s42"])
    p.add_argument("--blend", action="store_true", help="évalue aussi le mélange figé forêt + séquentiel")
    a = p.parse_args()

    stream, lab = load_stream()
    starts = attempt_starts(lab["attempt"], np.asarray(stream.t))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    frozen = json.loads((RESULTS / "val_blend_hist.json").read_text()) if a.blend else None
    forest_raw = load_forest()[0] if a.blend else None
    X = load_features()[0] if a.blend else None
    for name in a.models:
        z = {"test": logits(name, "test"), "tail": logits(name, "tail")}
        z["test_lit"] = np.concatenate([z["test"], z["tail"]])  # 09/09 → fin = test puis traîne
        assert all(len(z[sp]) == np.subtract(*stream.split_range(sp)[::-1]) for sp in SPLITS)
        thr_name = threshold(name)
        # Le seuil a été choisi sur des probabilités : on compare des probabilités (pas d'arrondi au seuil).
        prob = {sp: 1.0 / (1.0 + np.exp(-v)) for sp, v in z.items()}
        res = assess(name, prob, thr_name, stream, lab, starts)
        seed = int(name.rsplit("_s", 1)[1])
        write_result(f"eval_hist_{name}_{'-'.join(SPLITS)}_{stamp}", {"model": name, "threshold": thr_name,
                                                                      "splits": res}, seed=seed)  # fmt: skip
        if frozen and name == frozen["model"]:
            w, thr = frozen["full_val"]["w"], frozen["full_val"]["threshold"]
            s = {sp: w * forest_raw(np.asarray(X[slice(*stream.split_range(sp))], dtype=np.float32))
                 + (1 - w) * z[sp] for sp in SPLITS}  # fmt: skip
            res = assess(f"mélange w={w}", s, thr, stream, lab, starts)
            write_result(f"eval_blend_hist_{'-'.join(SPLITS)}_{stamp}", {"model": name, "w": w, "threshold": thr,
                                                                         "splits": res}, seed=seed)  # fmt: skip


if __name__ == "__main__":
    main()
