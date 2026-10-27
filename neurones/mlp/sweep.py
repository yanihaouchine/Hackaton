"""Budget d'hyperparamètres fixé à l'avance : grille sur la seed 42, puis 5 seeds pour la meilleure configuration.

Critère de choix : PR-AUC de validation (indépendante du seuil). Le test n'est jamais lu.
Usage (depuis neurones/) : python -m mlp.sweep  ->  results/val_mlp_sweep.json
"""

from __future__ import annotations

import numpy as np

from mlp.common import write_result
from mlp.train import run

GRID = [(hidden, neg) for hidden in ((64, 32), (128, 64), (256, 128)) for neg in (0.25, 1.0)]
SEEDS = (42, 43, 44, 45, 46)


def tag(hidden: tuple[int, ...], neg: float, seed: int) -> str:
    return f"h{'-'.join(map(str, hidden))}_n{int(neg * 100)}_s{seed}"


def summary(p: dict) -> dict:
    v = p["val"]
    return {"name": p["name"], "pr_auc": v["pr_auc"], "f1": v["f1"], "precision": v["precision"],
            "recall": v["recall"], "best_epoch": p["best_epoch"], "train_seconds": p["seconds"]["train"],
            "n_params": p["n_params"]}  # fmt: skip


def main() -> None:
    grid = []
    for hidden, neg in GRID:
        print(f"grille : {tag(hidden, neg, 42)}", flush=True)
        grid.append({"hidden": list(hidden), "neg_rate": neg, **summary(run(hidden, neg, 42, tag(hidden, neg, 42)))})
    best = max(grid, key=lambda g: g["pr_auc"])
    hidden, neg = tuple(best["hidden"]), best["neg_rate"]

    seeds = [summary(run(hidden, neg, s, tag(hidden, neg, s))) if s != 42 else best for s in SEEDS]
    agg = {m: {"mean": float(np.mean([r[m] for r in seeds])), "std": float(np.std([r[m] for r in seeds], ddof=1))}
           for m in ("pr_auc", "f1", "precision", "recall")}  # fmt: skip
    path = write_result("val_mlp_sweep", {"criterion": "val pr_auc", "grid": grid,
                                          "best": {"hidden": list(hidden), "neg_rate": neg},
                                          "seeds": seeds, "seeds_mean_std": agg}, seed=42)  # fmt: skip
    print(f"meilleure configuration : {tag(hidden, neg, 42)[:-4]} ; {path}")
    for m, d in agg.items():
        print(f"  {m:10s} {d['mean']:.4f} ± {d['std']:.4f}")


if __name__ == "__main__":
    main()
