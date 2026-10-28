"""Étape 3 : MLP v2 + réseau récurrent qui lit l'historique brut des comptes (8 dernières transactions de
l'émetteur et du destinataire, tous formats). Configuration fixée d'avance : celle du MLP v2 (64→32, 100 % des
licites, AdamW, lots de 4 096, arrêt précoce sur la PR-AUC de validation, patience 5), GRU de 16 unités partagé
entre émetteur et destinataire. Le test n'est jamais lu ici.
Usage (depuis neurones/) : python -m mlp.train_hist --seed 42  ->  results/val_hist_<nom>.json, artifacts/<nom>/
"""

from __future__ import annotations

import argparse
import json
import math
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from mlp.common import ARTIFACTS, best_f1_threshold, evaluate, load_features, load_stream, pr_auc, write_result
from mlp.history import EVENT_DIM, K, batch, build_index
from mlp.prep import Prep, fit_prep
from mlp.train import KS, THREADS

FIELDS = ("t", "src", "dst", "usd_paid", "fmt", "cur_paid", "cur_recv")


class SeqEncoder(nn.Module):
    """Résume une séquence d'événements passés (du plus ancien au plus récent) en un vecteur."""

    def __init__(self, hidden: int):
        super().__init__()
        self.gru = nn.GRU(EVENT_DIM, hidden, batch_first=True)

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(torch.flip(ev, dims=[1]))  # historique stocké du plus récent au plus ancien
        return h[0]


class HistNet(nn.Module):
    def __init__(self, prep: Prep, hidden: tuple[int, ...], gru: int, dropout: float):
        super().__init__()
        self.prep, self.hidden, self.gru_size, self.dropout = prep, tuple(hidden), gru, dropout
        self.enc = SeqEncoder(gru)
        layers, d = [], prep.out_dim + 2 * gru + 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, es, ms, ed, md) -> torch.Tensor:
        z = torch.cat([self.prep(x), self.enc(es), self.enc(ed), ms.float().mean(1, keepdim=True),
                       md.float().mean(1, keepdim=True)], dim=1)  # fmt: skip
        return self.mlp(z).squeeze(1)


def tensors(X, cols, hist, rows, stream_ram):
    x = torch.from_numpy(np.asarray(X[rows][:, cols], dtype=np.float32))
    es, ms, ed, md = batch(*hist, rows, stream_ram)
    return x, torch.from_numpy(es), torch.from_numpy(ms), torch.from_numpy(ed), torch.from_numpy(md)


def logits(net, X, cols, hist, rows, stream_ram, size: int = 16384) -> np.ndarray:
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), size):
            out.append(net(*tensors(X, cols, hist, rows[i : i + size], stream_ram)).numpy())
    return np.concatenate(out)


def run(seed: int, name: str, hidden=(64, 32), gru: int = 16, neg_rate: float = 1.0, dropout: float = 0.1,
        lr: float = 1e-3, weight_decay: float = 1e-4, size: int = 4096, max_epochs: int = 30,
        patience: int = 5) -> dict:  # fmt: skip
    from aml.train import subsample_rows

    torch.manual_seed(seed)
    torch.set_num_threads(THREADS)
    torch.use_deterministic_algorithms(True)
    t_start = time.perf_counter()
    X, all_names, cols = load_features()
    stream, lab = load_stream()
    stream_ram = SimpleNamespace(**{f: np.asarray(getattr(stream, f)) for f in FIELDS})  # ≈ 100 Mo
    hist = build_index(stream_ram.src, stream_ram.dst, K)
    y = lab["y"]
    names = [all_names[i] for i in cols]
    tr_lo, tr_hi = stream.split_range("train")
    va_lo, va_hi = stream.split_range("val")
    rows = subsample_rows(y, tr_lo, tr_hi, neg_rate, seed)
    va_rows = np.arange(va_lo, va_hi)
    y_va = y[va_lo:va_hi]
    prep = fit_prep(np.asarray(X[rows][:, cols], dtype=np.float32), names)
    net = HistNet(prep, hidden, gru, dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed)
    y_t = torch.from_numpy(y.astype(np.float32))

    best, best_state, wait, history = -1.0, None, 0, []
    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        net.train()
        perm = rows[torch.randperm(len(rows), generator=gen).numpy()]
        total = 0.0
        for i in range(0, len(perm), size):
            b = np.sort(perm[i : i + size])  # tri : lectures mmap plus locales, sans effet sur la perte
            opt.zero_grad()
            loss = loss_fn(net(*tensors(X, cols, hist, b, stream_ram)), y_t[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        ap = pr_auc(y_va, logits(net, X, cols, hist, va_rows, stream_ram))
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_pr_auc": ap,
                        "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip
        print(f"  époque {epoch:2d}  perte {total / len(perm):.5f}  PR-AUC val {ap:.4f}  "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)  # fmt: skip
        if ap > best:
            best, wait = ap, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    t_train = time.perf_counter() - t0
    net.load_state_dict(best_state)
    offset = math.log(neg_rate)
    s_va = 1.0 / (1.0 + np.exp(-(logits(net, X, cols, hist, va_rows, stream_ram) + offset)))
    thr = best_f1_threshold(y_va, s_va)
    val = evaluate(y_va, s_va, thr["threshold"], np.asarray(stream.t[va_lo:va_hi]), lab["attempt"][va_lo:va_hi],
                   lab["typ"][va_lo:va_hi], ks=KS)  # fmt: skip
    out = ARTIFACTS / name
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "names": names, "cat_sizes": prep.cat_sizes, "hidden": list(hidden),
                "gru": gru, "dropout": dropout, "neg_rate": neg_rate, "k": K}, out / "model.pt")  # fmt: skip
    (out / "meta.json").write_text(json.dumps({"threshold": thr["threshold"], "neg_rate": neg_rate, "k": K}) + "\n")
    payload = {
        "name": name,
        "split": "val",
        "config": {
            "hidden": list(hidden),
            "gru": gru,
            "k": K,
            "neg_rate": neg_rate,
            "dropout": dropout,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch": size,
            "max_epochs": max_epochs,
            "patience": patience,
            "threads": THREADS,
        },  # fmt: skip
        "n_params": int(sum(p.numel() for p in net.parameters())),
        "best_epoch": int(max(history, key=lambda h: h["val_pr_auc"])["epoch"]),
        "history": history,
        "seconds": {"train": round(t_train, 1), "total": round(time.perf_counter() - t_start, 1)},
        "val": val,
    }
    path = write_result(f"val_hist_{name}", payload, seed)
    print(f"{name} : PR-AUC val {val['pr_auc']:.4f}, F1 val {val['f1']:.4f} (seuil {thr['threshold']:.4g}) -> {path}")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--name")
    a = p.parse_args()
    run(a.seed, a.name or f"hist_h64-32_g16_s{a.seed}", max_epochs=a.max_epochs)


if __name__ == "__main__":
    main()
