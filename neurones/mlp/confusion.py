"""Matrices de confusion du MLP livré et de la forêt strict de arbre/, chacun à son seuil de validation.

Repris de la v1 de neurones/ (commit 0c12204). Le test n'est inclus qu'avec --test, et seulement si
l'évaluation figée du modèle (mlp.evaluate) a déjà été faite. La forêt est rejouée sur cette machine et
recoupée avec les chiffres publiés par arbre/.
Usage (depuis neurones/) : python -m mlp.confusion [--test]  ->  results/confusion.json
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort

from mlp.common import ARTIFACTS, REPO, RESULTS, load_features, load_stream, write_result

FOREST_TEST = REPO / "arbre" / "results" / "eval_strict_test-tail-test_lit_20261027-065943.json"


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
    from aml.forest import c_raw
    from aml.model import load_bundle

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    p.add_argument("--test", action="store_true", help="après l'évaluation figée du MLP uniquement")
    args = p.parse_args()
    if args.test and not list(RESULTS.glob(f"eval_mlp_{args.model}_*.json")):
        sys.exit("Le test de ce modèle n'a pas encore été évalué (mlp.evaluate) : on ne le regarde pas ici.")

    meta = json.loads((ARTIFACTS / args.model / "meta.json").read_text())
    thr_mlp = meta["threshold"]
    sess = ort.InferenceSession(str(ARTIFACTS / args.model / "model.onnx"), providers=["CPUExecutionProvider"])
    bundle = load_bundle("strict")
    X, _, cols = load_features()
    stream, lab = load_stream()
    y_all = lab["y"]

    out = {"models": {"mlp": {"name": args.model, "threshold": thr_mlp}, "forest": {"threshold": bundle.threshold}}}
    for split in ["val"] + (["test"] if args.test else []):
        lo, hi = stream.split_range(split)
        x_all = np.asarray(X[lo:hi], dtype=np.float32)  # la forêt lit les 104 colonnes du moteur
        x = np.ascontiguousarray(x_all[:, cols])
        y = y_all[lo:hi]
        p_mlp = np.concatenate([sess.run(None, {"features": x[i : i + 65536]})[0] for i in range(0, len(x), 65536)])
        p_forest = 1.0 / (1.0 + np.exp(-c_raw(bundle.forest, x_all)))
        out[split] = {"n_tx": len(y), "mlp": matrix(y, p_mlp, thr_mlp), "forest": matrix(y, p_forest, bundle.threshold)}

    f = out["val"]["forest"]
    pub = bundle.info["val_at_threshold"]
    out["forest_check_val"] = {"published": {k: pub[k] for k in ("tp", "fp", "fn")},
                               "recomputed": [f["VP"], f["FP"], f["FN"]]}  # fmt: skip
    if args.test:
        pub = json.loads(FOREST_TEST.read_text(encoding="utf-8"))["splits"]["test"]
        f = out["test"]["forest"]
        out["forest_check_test"] = {"published": {k: pub[k] for k in ("tp", "fp", "fn")},
                                    "recomputed": [f["VP"], f["FP"], f["FN"]]}  # fmt: skip
    write_result("confusion", out, seed=42)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
