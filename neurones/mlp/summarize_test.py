"""Agrège les évaluations figées (dernier fichier eval_mlp_<modèle>_* de chaque seed) : moyenne ± écart-type.

Ne relit pas le test : lit seulement les fichiers écrits par mlp.evaluate.
Usage (depuis neurones/) : python -m mlp.summarize_test  ->  results/test_mlp_seeds.json
"""

from __future__ import annotations

import json

import numpy as np

from mlp.common import RESULTS, write_result

CONFIG = "h64-32_n100"
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("f1", "precision", "recall", "pr_auc", "n_alerts")


def main() -> None:
    files = [sorted(RESULTS.glob(f"eval_mlp_{CONFIG}_s{s}_*.json"))[-1] for s in SEEDS]
    runs = [json.loads(f.read_text()) for f in files]
    out = {}
    for sp in runs[0]["splits"]:
        vals = {m: [r["splits"][sp][m] for r in runs] for m in METRICS}
        out[sp] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
                   for m, v in vals.items()}  # fmt: skip
        print(f"{sp:8s} F1 {out[sp]['f1']['mean']:.4f} ± {out[sp]['f1']['std']:.4f}  "
              f"PR-AUC {out[sp]['pr_auc']['mean']:.4f} ± {out[sp]['pr_auc']['std']:.4f}")  # fmt: skip
    write_result("test_mlp_seeds", {"config": CONFIG, "sources": [f.name for f in files], "splits": out}, seed=42)


if __name__ == "__main__":
    main()
