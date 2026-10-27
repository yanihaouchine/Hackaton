"""Entraînement d'un MLP sur le train, arrêt précoce et seuil sur la validation ; le test n'est jamais lu ici.

Usage (depuis neurones/) : python -m mlp.train --hidden 64,32 --neg-rate 0.25 --seed 42
Écrit results/val_mlp_<nom>.json et artifacts/<nom>/ (model.onnx, meta.json).
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

from mlp.common import (
    ARTIFACTS,
    FEATURE_DIGEST,
    best_f1_threshold,
    evaluate,
    load_features,
    load_stream,
    pr_auc,
    subsample_rows,
    write_result,
)
from mlp.prep import Prep, fit_prep

THREADS = 4
KS = (10, 25, 50, 100, 200, 500)


class Net(nn.Module):
    """Prétraitement figé puis MLP ; renvoie le logit appris sur le train sous-échantillonné."""

    def __init__(self, prep: Prep, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        self.prep = prep
        self.hidden, self.dropout = tuple(hidden), dropout
        layers, d = [], prep.out_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.prep(x)).squeeze(1)


class Scorer(nn.Module):
    """Modèle exporté : probabilité recalibrée à la proportion réelle (cote × neg_rate)."""

    def __init__(self, net: Net, neg_rate: float):
        super().__init__()
        self.net = net
        self.offset = math.log(neg_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x) + self.offset)


def logits(net: Net, x: np.ndarray, batch: int = 65536) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            out.append(net(torch.from_numpy(np.ascontiguousarray(x[lo : lo + batch]))).numpy())
    return np.concatenate(out)


def run(hidden: tuple[int, ...], neg_rate: float, seed: int, name: str, dropout: float = 0.1, lr: float = 1e-3,
        weight_decay: float = 1e-4, batch: int = 4096, max_epochs: int = 30, patience: int = 5) -> dict:  # fmt: skip
    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    t_start = time.perf_counter()

    X, all_names, cols = load_features()
    stream, lab = load_stream()
    y = lab["y"]
    names = [all_names[i] for i in cols]
    tr_lo, tr_hi = stream.split_range("train")
    va_lo, va_hi = stream.split_range("val")
    rows = subsample_rows(y, tr_lo, tr_hi, neg_rate, seed)  # mêmes lignes que arbre/ à seed et taux égaux
    x_tr = np.asarray(X[rows][:, cols], dtype=np.float32)
    y_tr = torch.from_numpy(y[rows].astype(np.float32))
    x_va = np.asarray(X[va_lo:va_hi][:, cols], dtype=np.float32)
    y_va = y[va_lo:va_hi]
    t_load = time.perf_counter() - t_start

    net = Net(fit_prep(x_tr, names), hidden, dropout)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    xt = torch.from_numpy(x_tr)

    best, best_state, wait, history = -1.0, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for lo in range(0, len(perm), batch):
            b = perm[lo : lo + batch]
            opt.zero_grad()
            loss = loss_fn(net(xt[b]), y_tr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        ap = pr_auc(y_va, logits(net, x_va))
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_pr_auc": ap})
        print(f"  époque {epoch:2d}  perte {total / len(perm):.5f}  PR-AUC val {ap:.4f}", flush=True)
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    t_train = time.perf_counter() - t0
    net.load_state_dict(best_state)
    scorer = Scorer(net, neg_rate).eval()

    with torch.no_grad():
        s_va = torch.sigmoid(torch.from_numpy(logits(net, x_va)) + scorer.offset).numpy()
    thr = best_f1_threshold(y_va, s_va)
    t_va = stream.t[va_lo:va_hi]
    val = evaluate(y_va, s_va, thr["threshold"], t_va, lab["attempt"][va_lo:va_hi], lab["typ"][va_lo:va_hi], ks=KS)

    export = export_onnx(scorer, names, x_va[:20000], name, thr["threshold"], neg_rate)
    payload = {
        "name": name,
        "split": "val",
        "config": {
            "hidden": list(hidden),
            "neg_rate": neg_rate,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch": batch,
            "max_epochs": max_epochs,
            "patience": patience,
            "threads": THREADS,
        },  # fmt: skip
        "data": {
            "train_rows": int(len(rows)),
            "train_pos": int(y[rows].sum()),
            "val_rows": int(len(y_va)),
            "n_inputs_raw": len(names),
            "n_inputs_net": net.prep.out_dim,
        },  # fmt: skip
        "n_params": int(n_params),
        "best_epoch": int(max(history, key=lambda h: h["val_pr_auc"])["epoch"]),
        "history": history,
        "seconds": {
            "load": round(t_load, 1),
            "train": round(t_train, 1),
            "total": round(time.perf_counter() - t_start, 1),
        },
        "val": val,
        "export": export,
    }
    path = write_result(f"val_mlp_{name}", payload, seed)
    print(f"{name} : PR-AUC val {val['pr_auc']:.4f}, F1 val {val['f1']:.4f} (seuil {thr['threshold']:.4g}) -> {path}")
    return payload


def load_scorer(name: str) -> tuple[Scorer, dict]:
    """Modèle torch livré (artifacts/<nom>/model.pt) et ses métadonnées (seuil, features)."""
    ck = torch.load(ARTIFACTS / name / "model.pt", weights_only=True)
    n_dense = ck["state_dict"]["prep.mean"].numel()
    prep = Prep(ck["names"], ck["cat_sizes"], np.zeros(n_dense), np.ones(n_dense))
    net = Net(prep, tuple(ck["hidden"]), ck["dropout"])
    net.load_state_dict(ck["state_dict"])
    meta = json.loads((ARTIFACTS / name / "meta.json").read_text())
    return Scorer(net, ck["neg_rate"]).eval(), meta


def export_onnx(scorer: Scorer, names: list[str], x_check: np.ndarray, name: str, threshold: float,
                neg_rate: float) -> dict:  # fmt: skip
    """Exporte prétraitement + réseau en ONNX et vérifie l'écart avec torch."""
    import onnxruntime as ort

    out = ARTIFACTS / name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "model.onnx"
    torch.onnx.export(scorer, (torch.from_numpy(x_check[:2]),), str(path), input_names=["features"],
                      output_names=["score"], dynamic_axes={"features": {0: "n"}, "score": {0: "n"}},
                      dynamo=False)  # fmt: skip
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    s_ort = sess.run(None, {"features": x_check})[0]
    with torch.no_grad():
        s_torch = scorer(torch.from_numpy(x_check)).numpy()
    meta = {"feature_digest": FEATURE_DIGEST, "feature_names": names, "threshold": threshold, "neg_rate": neg_rate}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    # Poids torch : nécessaires aux explications (gradient × entrée), pas à l'inférence.
    net = scorer.net
    ck = {"state_dict": net.state_dict(), "names": names, "cat_sizes": net.prep.cat_sizes,
          "hidden": list(net.hidden), "dropout": net.dropout, "neg_rate": neg_rate}  # fmt: skip
    torch.save(ck, out / "model.pt")
    return {"path": str(path.relative_to(ARTIFACTS.parent)), "bytes": path.stat().st_size,
            "max_abs_diff_ort_torch": float(np.max(np.abs(s_ort - s_torch)))}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hidden", default="64,32")
    p.add_argument("--neg-rate", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--name")
    a = p.parse_args()
    hidden = tuple(int(h) for h in a.hidden.split(","))
    name = a.name or f"h{'-'.join(map(str, hidden))}_n{int(a.neg_rate * 100)}_s{a.seed}"
    run(hidden, a.neg_rate, a.seed, name, max_epochs=a.max_epochs)


if __name__ == "__main__":
    main()
