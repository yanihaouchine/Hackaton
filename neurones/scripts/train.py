"""Entraîne le MLP : poids de la classe illicite choisi sur la validation, arrêt précoce sur la PR-AUC de validation.
Le test n'est jamais lu ici.
Usage (depuis neurones/) : ../.venv/bin/python scripts/train.py [--seeds 42,43,44] [--pos-weights 1,10,100]
Sorties : artifacts/mlp.pt (modèle retenu, première seed), results/train_mlp.json (courbes, grille, seeds)
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn

from nn.data import RESULTS, SEED, fit_preprocessing, load, rows_of, train_rows
from nn.model import mlp, n_params

THREADS = 4
BATCH = 4096
LR = 1e-3
MAX_EPOCHS = 40
PATIENCE = 3
ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"


def predict_logits(net: nn.Module, X: np.ndarray, batch: int = 65536) -> np.ndarray:
    net.eval()
    with torch.no_grad():
        chunks = [net(torch.from_numpy(X[i : i + batch])).squeeze(1).numpy() for i in range(0, len(X), batch)]
        return np.concatenate(chunks)


def fit(X_tr, y_tr, X_va, y_va, pos_weight: float, seed: int) -> tuple[nn.Module, dict]:
    """Une course d'entraînement ; renvoie le réseau de la meilleure epoch (PR-AUC de validation) et ses courbes."""
    from aml.evaluate import pr_auc

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = mlp(X_tr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    yv = torch.from_numpy(y_va.astype(np.float32))
    curves, best, best_state, wait = [], -1.0, None, 0
    t0 = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(len(X_tr)))
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i : i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(xt[b]).squeeze(1), yt[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        s_va = predict_logits(net, X_va)
        val_loss = float(loss_fn(torch.from_numpy(s_va), yv))
        ap = pr_auc(y_va, s_va)
        curves.append({"epoch": epoch, "train_loss": total / len(X_tr), "val_loss": val_loss, "val_pr_auc": ap})
        print(f"  epoch {epoch:2d}  loss train {curves[-1]['train_loss']:.4f}  val {val_loss:.4f}  PR-AUC val {ap:.4f}")
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    net.load_state_dict(best_state)
    best_epoch = max(curves, key=lambda c: c["val_pr_auc"])["epoch"]
    info = {"seed": seed, "pos_weight": pos_weight, "best_epoch": best_epoch, "epochs_run": len(curves),
            "val_pr_auc": best, "train_seconds": round(time.perf_counter() - t0, 1), "curves": curves}  # fmt: skip
    return net, info


def main() -> None:
    from aml.evaluate import best_f1_threshold
    from aml.provenance import run_info

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--pos-weights", default="1,10,100")
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    weights = [float(w) for w in a.pos_weights.split(",")]
    torch.set_num_threads(THREADS)

    ds = load()
    names = [ds.names[i] for i in ds.cols]
    x_va, y_va = rows_of(ds, ds.ranges["val"])

    def data(seed: int):
        x_tr, y_tr = rows_of(ds, train_rows(ds, seed))
        pre = fit_preprocessing(x_tr, names)
        return pre, pre.transform(x_tr), y_tr, pre.transform(x_va)

    # 1. Poids de la classe illicite choisi sur la validation (seed 42)
    pre, X_tr, y_tr, X_va = data(SEED)
    grid = []
    for w in weights:
        print(f"pos_weight = {w:g}", flush=True)
        _, info = fit(X_tr, y_tr, X_va, y_va, w, SEED)
        grid.append(info)
    chosen = max(grid, key=lambda r: r["val_pr_auc"])["pos_weight"]

    # 2. Configuration retenue sur plusieurs seeds (variance) ; le modèle livré est celui de la première seed
    runs, kept = [], None
    for seed in seeds:
        print(f"seed {seed} (pos_weight = {chosen:g})", flush=True)
        if seed != SEED:
            pre_s, X_tr, y_tr, X_va_s = data(seed)
        else:
            pre_s, X_va_s = pre, X_va
        net, info = fit(X_tr, y_tr, X_va_s, y_va, chosen, seed)
        thr = best_f1_threshold(y_va, predict_logits(net, X_va_s))
        info["val_best_f1"] = thr
        runs.append(info)
        if kept is None:
            kept = (net, pre_s, thr)

    net, pre_kept, thr = kept
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "preprocessing": pre_kept.__dict__, "threshold_logit": thr["threshold"],
                "pos_weight": chosen, "seed": seeds[0]}, ARTIFACTS / "mlp.pt")  # fmt: skip
    f1s = [r["val_best_f1"]["f1"] for r in runs]
    aps = [r["val_pr_auc"] for r in runs]
    out = {
        "run": run_info(SEED),
        "features_digest": ds.features_digest,
        "n_inputs": pre_kept.n_inputs,
        "n_params": n_params(net),
        "hyper": {
            "hidden": [128, 64],
            "dropout": 0.1,
            "lr": LR,
            "batch": BATCH,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "threads": THREADS,
            "neg_rate": 0.25,
        },  # fmt: skip
        "pos_weight_grid": [{k: r[k] for k in ("pos_weight", "val_pr_auc", "best_epoch")} for r in grid],
        "pos_weight_chosen": chosen,
        "seeds": runs,
        "val_summary": {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "pr_auc_mean": float(np.mean(aps)),
            "pr_auc_std": float(np.std(aps)),
        },  # fmt: skip
        "kept": {"seed": seeds[0], "threshold_logit": thr["threshold"], "val_at_threshold": thr},
        "grid_curves": {str(r["pos_weight"]): r["curves"] for r in grid},
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "train_mlp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pos_weight_grid", "pos_weight_chosen", "val_summary", "kept")}, indent=2))


if __name__ == "__main__":
    main()
