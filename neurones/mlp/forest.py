"""Forêt d'arbres de référence pour le mélange : celle de arbre/ sur HI-Small, réentraînée ici pour une autre variante.

Sur HI-Small : forêt `strict` livrée par arbre/ (évaluateur C). Sur une autre variante, son modèle n'est pas
versionné : on le réentraîne avec les fonctions de arbre/ en lecture seule et sa méthode exacte (familles du
modèle strict, toutes les illicites + 25 % des licites, seed 42, arrêt précoce sur la PR-AUC de validation,
seuil au F1 maximal de validation), et on l'écrit dans artifacts/<variante>/forest/, jamais dans arbre/.
Usage (depuis neurones/) : AML_VARIANT=LI-Small python -m mlp.forest  ->  results/<variante>/val_forest.json
"""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np

from mlp.common import ARTIFACTS, VARIANT, best_f1_threshold, load_features, load_stream, pr_auc, write_result

SEED, NEG_RATE = 42, 0.25


def load_forest() -> tuple[Callable[[np.ndarray], np.ndarray], float, str]:
    """(log-cote brute à partir des 104 colonnes du moteur, seuil en probabilité, nom)."""
    if VARIANT == "HI-Small":
        from aml.forest import c_raw
        from aml.model import load_bundle

        b = load_bundle("strict")
        return (lambda x: c_raw(b.forest, x).astype(np.float64)), b.threshold, "strict (arbre/)"
    import lightgbm as lgb

    d = ARTIFACTS / "forest"
    meta = json.loads((d / "meta.json").read_text())
    booster = lgb.Booster(model_file=str(d / "lgbm.txt"))
    cols = meta["cols"]
    return (lambda x: booster.predict(x[:, cols], raw_score=True)), meta["threshold"], f"strict réentraînée ({VARIANT})"


def main() -> None:
    from aml.train import ALL_FAMILIES, fit, select, subsample_rows

    if VARIANT == "HI-Small":
        raise SystemExit("Sur HI-Small, la forêt est celle livrée par arbre/ : rien à entraîner.")
    X, names, _ = load_features()
    stream, lab = load_stream()
    y = lab["y"]
    cols = select(names, ALL_FAMILIES)
    tr, va = stream.split_range("train"), stream.split_range("val")
    rows = subsample_rows(y, *tr, neg_rate=NEG_RATE, seed=SEED)
    booster, s_val, info = fit(X, y, names, rows, va, cols, seed=SEED)
    yv = y[va[0] : va[1]]
    thr = best_f1_threshold(yv, s_val)
    d = ARTIFACTS / "forest"
    d.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(d / "lgbm.txt"), num_iteration=booster.best_iteration)
    meta = {"variant": VARIANT, "cols": cols, "families": ALL_FAMILIES, "neg_rate": NEG_RATE, "seed": SEED,
            "best_iteration": int(booster.best_iteration), "threshold": thr["threshold"]}  # fmt: skip
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    payload = {"forest": meta, "train_rows": int(len(rows)), "train_seconds": info["train_seconds"],
               "val_pr_auc": pr_auc(yv, s_val), "val_best_f1": thr, "params": info["params"]}  # fmt: skip
    path = write_result("val_forest", payload, seed=SEED)
    print(f"forêt {VARIANT} : PR-AUC val {payload['val_pr_auc']:.4f}, F1 val {thr['f1']:.4f}, "
          f"{info['train_seconds']} s -> {path}")  # fmt: skip


if __name__ == "__main__":
    main()
