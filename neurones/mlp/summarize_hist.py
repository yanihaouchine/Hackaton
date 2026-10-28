"""Variance du modèle séquentiel sur 5 seeds (validation), à côté du MLP v2 : moyenne ± écart-type.

Lit seulement les fichiers val_hist_* et val_mlp_* déjà écrits ; ne relit aucune donnée.
Usage (depuis neurones/) : python -m mlp.summarize_hist  ->  results/val_hist_seeds.json
"""

from __future__ import annotations

import json

import numpy as np

from mlp.common import RESULTS, write_result

SEEDS = (42, 43, 44, 45, 46)
METRICS = ("pr_auc", "f1", "precision", "recall")
RUNS = {"hist": "val_hist_hist_h64-32_g16_s{}.json", "mlp_v2": "val_mlp_h64-32_n100_s{}.json"}


def main() -> None:
    out = {}
    for key, pattern in RUNS.items():
        runs = [json.loads((RESULTS / pattern.format(s)).read_text()) for s in SEEDS]
        out[key] = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v}
                    for m in METRICS for v in [[r["val"][m] for r in runs]]}  # fmt: skip
        out[key]["n_params"] = runs[0]["n_params"]
        print(f"{key:7s} PR-AUC {out[key]['pr_auc']['mean']:.4f} ± {out[key]['pr_auc']['std']:.4f}  "
              f"F1 {out[key]['f1']['mean']:.4f} ± {out[key]['f1']['std']:.4f}")  # fmt: skip
    write_result("val_hist_seeds", out, seed=42)


if __name__ == "__main__":
    main()
