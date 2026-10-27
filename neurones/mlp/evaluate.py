"""Évaluation d'un MLP figé (ONNX livré, seuil de validation) sur test, traîne et protocole littérature.

Le test n'est lu qu'ici, une fois par configuration figée ; chaque lecture écrit un fichier horodaté (jamais écrasé).
Même protocole que arbre/ (aml.experiments.evaluate_model) : délais depuis le vrai début des tentatives, niveau cas.
Usage (depuis neurones/) : python -m mlp.evaluate --models h64-32_n100_s42 [h64-32_n100_s43 ...]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np
import onnxruntime as ort

from mlp.common import ARTIFACTS, FEATURE_DIGEST, evaluate, load_features, load_stream, write_result

SPLITS = ("test", "tail", "test_lit")
KS = (10, 25, 50, 100, 200, 500)


def attempt_starts(attempt: np.ndarray, t: np.ndarray) -> dict[int, int]:
    """Début réel de chaque tentative, toutes périodes confondues."""
    m = attempt >= 0
    a, tt = attempt[m], t[m]
    out: dict[int, int] = {}
    for k, v in zip(a.tolist(), tt.tolist(), strict=True):
        out[k] = min(out.get(k, v), v)
    return out


def score(sess: ort.InferenceSession, x: np.ndarray, batch: int = 65536) -> np.ndarray:
    return np.concatenate([sess.run(None, {"features": x[i : i + batch]})[0] for i in range(0, len(x), batch)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--models", nargs="+", required=True)
    a = p.parse_args()

    from aml.cases import build_cases
    from aml.evaluate import case_report

    X, all_names, cols = load_features()
    stream, lab = load_stream()
    y, t_all = lab["y"], np.asarray(stream.t)
    starts = attempt_starts(lab["attempt"], t_all)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4

    for name in a.models:
        meta = json.loads((ARTIFACTS / name / "meta.json").read_text())
        if meta["feature_digest"] != FEATURE_DIGEST or meta["feature_names"] != [all_names[i] for i in cols]:
            raise ValueError(f"{name} : le modèle ne correspond pas aux features calculées")
        sess = ort.InferenceSession(str(ARTIFACTS / name / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        thr = meta["threshold"]
        out = {}
        for sp in SPLITS:
            lo, hi = stream.split_range(sp)
            s = score(sess, np.asarray(X[lo:hi][:, cols], dtype=np.float32))
            t = t_all[lo:hi]
            res = evaluate(y[lo:hi], s, thr, t, lab["attempt"][lo:hi], lab["typ"][lo:hi], ks=KS, attempt_start=starts)
            al = np.flatnonzero(s >= thr)
            ia = lo + al
            cases = build_cases(ia, np.asarray(stream.src[ia]), np.asarray(stream.dst[ia]), t[al],
                                np.asarray(stream.usd_paid[ia], dtype=np.float64), s[al])  # fmt: skip
            res["cases"] = case_report(cases, lambda idx: y[np.asarray(idx)] == 1)
            out[sp] = res
            print(f"{name} [{sp:8s}] P {res['precision']:.4f} R {res['recall']:.4f} F1 {res['f1']:.4f} "
                  f"IC95 {res['f1_ci95'][0]:.3f}-{res['f1_ci95'][1]:.3f} PR-AUC {res['pr_auc']:.4f} "
                  f"alertes {res['n_alerts']}", flush=True)  # fmt: skip
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed = int(name.rsplit("_s", 1)[1])
        write_result(f"eval_mlp_{name}_{'-'.join(SPLITS)}_{stamp}", {"model": name, "threshold": thr,
                                                                     "splits": out}, seed=seed)  # fmt: skip


if __name__ == "__main__":
    main()
