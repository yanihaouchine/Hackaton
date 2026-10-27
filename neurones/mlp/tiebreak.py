"""Départage 256→128 / 64→32 (100 % des licites) selon la règle fixée d'avance dans neurones/CLAUDE.md.

On garde 64→32 sauf si 256→128 le dépasse de plus d'un écart-type (5 seeds) en PR-AUC de validation.
L'écart-type de référence est le plus grand des deux (choix le plus exigeant pour le grand réseau).
Usage (depuis neurones/) : python -m mlp.tiebreak  ->  results/val_mlp_tiebreak.json
"""

from __future__ import annotations

import json

import numpy as np

from mlp.common import RESULTS, write_result

SEEDS = (42, 43, 44, 45, 46)
METRICS = ("pr_auc", "f1", "precision", "recall")


def collect(prefix: str) -> dict:
    runs = [json.loads((RESULTS / f"val_mlp_{prefix}_s{s}.json").read_text()) for s in SEEDS]
    out = {m: [r["val"][m] for r in runs] for m in METRICS}
    stats = {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": v} for m, v in out.items()}
    return {"n_params": runs[0]["n_params"], "onnx_bytes": runs[0]["export"]["bytes"],
            "train_seconds_mean": float(np.mean([r["seconds"]["train"] for r in runs])), "val": stats}  # fmt: skip


def main() -> None:
    small, large = collect("h64-32_n100"), collect("h256-128_n100")
    gap = large["val"]["pr_auc"]["mean"] - small["val"]["pr_auc"]["mean"]
    ref_std = max(large["val"]["pr_auc"]["std"], small["val"]["pr_auc"]["std"])
    winner = "h256-128_n100" if gap > ref_std else "h64-32_n100"
    path = write_result("val_mlp_tiebreak", {"rule": __doc__.splitlines()[2], "small": small, "large": large,
                                             "pr_auc_gap": gap, "reference_std": ref_std, "winner": winner},
                        seed=42)  # fmt: skip
    for name, r in (("64→32", small), ("256→128", large)):
        v = r["val"]
        print(f"{name:8s} params {r['n_params']:6d}  PR-AUC {v['pr_auc']['mean']:.4f} ± {v['pr_auc']['std']:.4f}"
              f"  F1 {v['f1']['mean']:.4f} ± {v['f1']['std']:.4f}  train {r['train_seconds_mean']:.0f} s")  # fmt: skip
    print(f"écart PR-AUC {gap:+.4f}, écart-type de référence {ref_std:.4f} -> {winner} ; {path}")


if __name__ == "__main__":
    main()
