"""Évaluation du mélange figé forêt + MLP (option A) sur test, traîne et protocole littérature.

Poids et seuil lus dans results/val_blend.json (choisis sur la validation seulement) ; jamais modifiés ici.
Même protocole que mlp.evaluate (délais depuis le vrai début des tentatives, niveau cas). Chaque lecture écrit
un fichier horodaté. Usage (depuis neurones/) : python -m mlp.evaluate_blend
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import onnxruntime as ort

from mlp.blend import logit
from mlp.common import ARTIFACTS, RESULTS, evaluate, load_features, load_stream, write_result
from mlp.evaluate import KS, SPLITS, attempt_starts, score
from mlp.forest import load_forest


def main() -> None:
    from aml.cases import build_cases
    from aml.evaluate import case_report

    frozen = json.loads((RESULTS / "val_blend.json").read_text())
    model, w, thr = frozen["model"], frozen["full_val"]["w"], frozen["full_val"]["threshold"]
    X, _, cols = load_features()
    stream, lab = load_stream()
    y, t_all = lab["y"], np.asarray(stream.t)
    starts = attempt_starts(lab["attempt"], t_all)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(ARTIFACTS / model / "model.onnx"), opts, providers=["CPUExecutionProvider"])
    forest_raw, _, forest_name = load_forest()

    out = {}
    for sp in SPLITS:
        lo, hi = stream.split_range(sp)
        x_all = np.asarray(X[lo:hi], dtype=np.float32)
        z_mlp = logit(score(sess, np.ascontiguousarray(x_all[:, cols])))
        s = w * forest_raw(x_all) + (1 - w) * z_mlp
        t = t_all[lo:hi]
        res = evaluate(y[lo:hi], s, thr, t, lab["attempt"][lo:hi], lab["typ"][lo:hi], ks=KS, attempt_start=starts)
        al = np.flatnonzero(s >= thr)
        ia = lo + al
        prio = 1.0 / (1.0 + np.exp(-s[al]))  # priorité des cas : log-cote mélangée ramenée dans [0, 1]
        cases = build_cases(ia, np.asarray(stream.src[ia]), np.asarray(stream.dst[ia]), t[al],
                            np.asarray(stream.usd_paid[ia], dtype=np.float64), prio)  # fmt: skip
        res["cases"] = case_report(cases, lambda idx: y[np.asarray(idx)] == 1)
        out[sp] = res
        print(f"mélange w={w} [{sp:8s}] P {res['precision']:.4f} R {res['recall']:.4f} F1 {res['f1']:.4f} "
              f"IC95 {res['f1_ci95'][0]:.3f}-{res['f1_ci95'][1]:.3f} PR-AUC {res['pr_auc']:.4f} "
              f"alertes {res['n_alerts']}", flush=True)  # fmt: skip
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    write_result(f"eval_blend_{'-'.join(SPLITS)}_{stamp}", {"model": model, "forest": forest_name, "w": w,
                                                            "threshold": thr, "splits": out}, seed=42)  # fmt: skip


if __name__ == "__main__":
    main()
