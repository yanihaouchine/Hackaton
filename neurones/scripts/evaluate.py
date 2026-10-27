"""Évaluation finale du MLP livré (ONNX) : test, traîne et protocole littérature, avec les fonctions de arbre/.
À lancer UNE SEULE FOIS, configuration figée (seuil fixé sur la validation). Refuse de réécrire le résultat.
Usage (depuis neurones/) : ../.venv/bin/python scripts/evaluate.py
Sortie : results/eval_mlp_test.json (à côté, les chiffres de la forêt strict de arbre/ pour comparaison)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nn.data import ARBRE, RESULTS, load, rows_of

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FOREST_RESULT = ARBRE / "results" / "eval_strict_test-tail-test_lit_20261027-065943.json"


def main() -> None:
    import onnxruntime as ort
    from aml.evaluate import binary_at, bootstrap_f1, pr_auc
    from aml.provenance import run_info

    out_path = RESULTS / "eval_mlp_test.json"
    if out_path.exists():
        sys.exit(f"{out_path} existe déjà : le test ne se regarde qu'une fois.")
    export = json.loads((RESULTS / "export_mlp.json").read_text(encoding="utf-8"))
    thr = export["threshold_proba"]
    sess = ort.InferenceSession(str(ARTIFACTS / "mlp.onnx"), providers=["CPUExecutionProvider"])

    ds = load()
    t_lo, _ = ds.ranges["test"]
    periods = {"test": ds.ranges["test"], "tail": ds.ranges["tail"], "test_lit": (t_lo, len(ds.y))}
    forest = json.loads(FOREST_RESULT.read_text(encoding="utf-8"))["splits"]
    splits, compare = {}, {}
    for name, rng in periods.items():
        x, y = rows_of(ds, rng)
        p = np.concatenate([sess.run(None, {"features": x[i : i + 65536]})[0] for i in range(0, len(x), 65536)])
        m = binary_at(y, p, thr)
        m.update(pr_auc=pr_auc(y, p), n_tx=len(y), n_pos=int(y.sum()))
        m["f1_ci95"] = list(np.percentile(bootstrap_f1(m["tp"], m["fp"], m["fn"], len(y)), [2.5, 97.5]))
        splits[name] = m
        f = forest[name]
        compare[name] = {k: {"mlp": m[k], "forest": f[k]} for k in ("precision", "recall", "f1", "pr_auc", "n_alerts")}
    out = {"run": run_info(), "threshold_proba": thr, "splits": splits, "compare_forest_strict": compare,
           "forest_source": str(FOREST_RESULT.relative_to(ARBRE.parent))}  # fmt: skip
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(compare, indent=2))


if __name__ == "__main__":
    main()
