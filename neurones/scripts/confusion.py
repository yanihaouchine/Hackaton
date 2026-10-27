"""Matrices de confusion du MLP et de la forêt strict de arbre/, même découpage, chacun à son seuil de validation.
Validation toujours ; test seulement après l'évaluation figée du MLP (scripts/evaluate.py), avec --test.
La forêt est rejouée sur cette machine et recoupée avec les chiffres publiés par arbre/.
Usage (depuis neurones/) : ../.venv/bin/python scripts/confusion.py [--test]
Sortie : results/confusion.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nn.data import ARBRE, RESULTS, load, rows_of

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
FOREST_TEST = ARBRE / "results" / "eval_strict_test-tail-test_lit_20261027-065943.json"


def matrix(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    """VP, FP, FN, VN au seuil, et les métriques calculées à partir d'eux seuls."""
    a, t = p >= thr, y == 1
    vp, fp, fn, vn = int(np.sum(a & t)), int(np.sum(a & ~t)), int(np.sum(~a & t)), int(np.sum(~a & ~t))
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    return {
        "VP": vp, "FP": fp, "FN": fn, "VN": vn, "threshold": thr,
        "precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "accuracy": (vp + vn) / len(y),
        "accuracy_always_licit": float(np.mean(~t)),  # modèle idiot : ne signale jamais rien
    }  # fmt: skip


def main() -> None:
    import onnxruntime as ort
    from aml.forest import c_raw
    from aml.model import load_bundle
    from aml.provenance import run_info

    a = argparse.ArgumentParser()
    a.add_argument("--test", action="store_true", help="après l'évaluation figée du MLP uniquement")
    args = a.parse_args()
    if args.test and not (RESULTS / "eval_mlp_test.json").exists():
        sys.exit("Le test du MLP n'a pas encore été évalué (scripts/evaluate.py) : on ne le regarde pas ici.")

    thr_mlp = json.loads((RESULTS / "export_mlp.json").read_text(encoding="utf-8"))["threshold_proba"]
    sess = ort.InferenceSession(str(ARTIFACTS / "mlp.onnx"), providers=["CPUExecutionProvider"])
    bundle = load_bundle("strict")
    ds = load()

    out = {"run": run_info(), "models": {"mlp": {"threshold": thr_mlp}, "forest": {"threshold": bundle.threshold}}}
    for split in ["val"] + (["test"] if args.test else []):
        lo, hi = ds.ranges[split]
        x, y = rows_of(ds, (lo, hi))
        p_mlp = np.concatenate([sess.run(None, {"features": x[i : i + 65536]})[0] for i in range(0, len(x), 65536)])
        x_all = np.asarray(ds.X[lo:hi])  # la forêt lit les 104 colonnes du moteur
        p_forest = 1.0 / (1.0 + np.exp(-c_raw(bundle.forest, x_all)))
        out[split] = {"n_tx": len(y), "mlp": matrix(y, p_mlp, thr_mlp), "forest": matrix(y, p_forest, bundle.threshold)}

    # Recoupement avec les chiffres publiés par arbre/ (même modèle, autre machine)
    f = out["val"]["forest"]
    pub = bundle.info["val_at_threshold"]
    out["forest_check_val"] = {
        "published": {k: pub[k] for k in ("tp", "fp", "fn")},
        "recomputed": [f["VP"], f["FP"], f["FN"]],
    }
    if args.test:
        pub = json.loads(FOREST_TEST.read_text(encoding="utf-8"))["splits"]["test"]
        f = out["test"]["forest"]
        out["forest_check_test"] = {"published": {k: pub[k] for k in ("tp", "fp", "fn")},
                                    "recomputed": [f["VP"], f["FP"], f["FN"]]}  # fmt: skip
    (RESULTS / "confusion.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
