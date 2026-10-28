"""Raisons d'une alerte du modèle séquentiel : features du moteur ET transactions passées des deux comptes.

Méthode « gradient × entrée » (celle du MLP v2), appliquée aux deux parties de l'entrée :
- features prétraitées : contribution regroupée par feature du moteur, puis phrases et rangs de `aml.explain` ;
- séquences : contribution de chaque transaction passée = somme de ∂logit/∂e × e sur ses caractéristiques
  (référence : aucune transaction, e = 0, comme le remplissage). Les textes sont écrits par le code : montants,
  délais et identifiants viennent du flux, jamais d'un modèle de langue.
Fidélité (validation, protocole de arbre/) : on retire les k raisons les plus fortes de chaque alerte (features
remplacées par la médiane du train, ou transactions passées effacées) et on compte les alertes qui repassent
sous le seuil ; témoin : k éléments tirés au hasard.
Usage (depuis neurones/) : python -m mlp.explain_hist [--model hist_h64-32_g16_s42]  ->  results/explain_hist.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mlp.common import VARIANT, write_result
from mlp.explain import prep_owners, to_engine_space
from mlp.history import batch
from mlp.train_hist import HistScorer


def contributions(net, x, es, ms, ed, md) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(features (n, colonnes du modèle), passé émetteur (n, K), passé destinataire (n, K)) : contributions au logit."""
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    z = net.prep(xt).detach().requires_grad_(True)
    a = torch.from_numpy(es).requires_grad_(True)
    b = torch.from_numpy(ed).requires_grad_(True)
    ms_t, md_t = torch.from_numpy(ms).float(), torch.from_numpy(md).float()
    h = torch.cat([z, net.enc(a), net.enc(b), ms_t.mean(1, keepdim=True), md_t.mean(1, keepdim=True)], dim=1)
    net.mlp(h).sum().backward()
    gz = (z.grad * z).detach().numpy()
    static = np.zeros((len(x), x.shape[1]))
    np.add.at(static.T, prep_owners(net.prep), gz.T)
    return static, (a.grad * a).sum(-1).detach().numpy(), (b.grad * b).sum(-1).detach().numpy()


def prob(sc: HistScorer, x, es, ms, ed, md) -> np.ndarray:
    with torch.no_grad():
        t = [torch.from_numpy(np.asarray(v)) for v in (x, es, ms, ed, md)]
        return torch.sigmoid(sc.net(*t) + sc.offset).numpy()


def describe_event(sc: HistScorer, names, row: int, past: int, side: str) -> str:
    """Phrase française d'une transaction passée, vue depuis l'opération `row`."""
    from aml.config import FORMATS
    from aml.explain import fmt_dur, fmt_usd

    r = sc.ram
    acc = r.src[row] if side == "src" else r.dst[row]
    other = r.dst[row] if side == "src" else r.src[row]
    who = "l'émetteur" if side == "src" else "le destinataire"
    out = r.src[past] == acc
    cp = r.dst[past] if out else r.src[past]
    if r.src[past] == r.dst[past]:
        what = f"auto-virement de {fmt_usd(float(r.usd_paid[past]))}"
    elif out:
        what = f"{who} a envoyé {fmt_usd(float(r.usd_paid[past]))} au compte {names[int(cp)]['key']}"
    else:
        what = f"{who} a reçu {fmt_usd(float(r.usd_paid[past]))} du compte {names[int(cp)]['key']}"
    extra = [FORMATS[int(r.fmt[past])]]
    if cp == other and r.src[past] != r.dst[past]:
        extra.append("même contrepartie que l'opération")
    if r.cur_paid[past] != r.cur_recv[past]:
        extra.append("change de devise")
    dt = float(r.t[row] - r.t[past])
    when = "dans la même minute" if dt == 0 else f"il y a {fmt_dur(dt)}"
    return f"{when} : {what} ({', '.join(extra)})"


def reasons(sc: HistScorer, bundle, names, countries, row: int, k: int = 5, k_hist: int = 3) -> dict:
    """Raisons d'une opération : features du moteur (format de arbre/) et transactions passées décisives."""
    from aml.explain import reasons as engine_reasons

    rows = np.array([row])
    x = np.ascontiguousarray(sc.X[rows][:, sc.cols], dtype=np.float32)
    es, ms, ed, md = batch(*sc.hist, rows, sc.ram)
    static, cs, cd = contributions(sc.net, x, es, ms, ed, md)
    x_engine = np.asarray(sc.X[row], dtype=np.float32)
    feats = engine_reasons(bundle, x_engine, to_engine_space(static, sc.cols, len(x_engine))[0], countries, k=k)
    past = [(float(cs[0, j]), "src", int(sc.hist[0][row, j])) for j in range(cs.shape[1]) if ms[0, j]]
    past += [(float(cd[0, j]), "dst", int(sc.hist[1][row, j])) for j in range(cd.shape[1]) if md[0, j]]
    past = [p for p in sorted(past, reverse=True) if p[0] > 0][:k_hist]
    hist = [{"side": s, "tx_index": i, "contribution": c, "text": describe_event(sc, names, row, i, s)}
            for c, s, i in past]  # fmt: skip
    pos_hist = float(np.clip(np.r_[cs[0], cd[0]], 0, None).sum())
    pos_feat = float(np.clip(static[0], 0, None).sum())
    return {"score": float(prob(sc, x, es, ms, ed, md)[0]), "features": feats, "history": hist,
            "share_history": pos_hist / max(pos_hist + pos_feat, 1e-12)}  # fmt: skip


def faithfulness(sc: HistScorer, rows: np.ndarray, median: np.ndarray, thr: float, k: int, seed: int = 42) -> dict:
    x = np.ascontiguousarray(sc.X[rows][:, sc.cols], dtype=np.float32)
    es, ms, ed, md = batch(*sc.hist, rows, sc.ram)
    before = prob(sc, x, es, ms, ed, md)
    static, cs, cd = contributions(sc.net, x, es, ms, ed, md)
    rng = np.random.default_rng(seed)
    r = np.arange(len(rows))[:, None]
    # 1. features : k premières remplacées par la médiane du train ; témoin : k features au hasard
    top = np.argsort(-static, axis=1)[:, :k]
    rnd = rng.random(static.shape).argsort(axis=1)[:, :k]
    out = {}
    for tag, idx in (("features", top), ("features_random", rnd)):
        xx = x.copy()
        xx[r, idx] = median[idx]
        out[tag] = float((prob(sc, xx, es, ms, ed, md) < thr).mean())
    # 2. historique : k transactions passées les plus fortes effacées (les deux comptes) ; témoin : k au hasard
    c_all = np.where(np.c_[ms, md], np.c_[cs, cd], -np.inf)
    noise = np.where(np.c_[ms, md], rng.random(c_all.shape), -np.inf)
    kk = int(min(k, np.c_[ms, md].sum(1).min())) if len(rows) else k
    for tag, score in (("history", c_all), ("history_random", noise)):
        idx = np.argsort(-score, axis=1)[:, :k]
        e = np.concatenate([es, ed], axis=1).copy()
        m = np.concatenate([ms, md], axis=1).copy()
        valid = np.take_along_axis(m, idx, 1)
        e[r, idx] = np.where(valid[..., None], 0.0, e[r, idx])
        m[r, idx] = False
        K = es.shape[1]
        out[tag] = float((prob(sc, x, e[:, :K], m[:, :K], e[:, K:], m[:, K:]) < thr).mean())
    return {"k": k, "n_alerts": int(len(rows)), "share_below_threshold_after": out, "min_valid_events": kk,
            "median_score_before": float(np.median(before))}  # fmt: skip


def main() -> None:
    from aml.data import AccountNames
    from aml.model import load_bundle

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="hist_h64-32_g16_s42")
    a = p.parse_args()
    if VARIANT != "HI-Small":
        raise SystemExit("Les phrases et rangs viennent du modèle strict de arbre/, livré pour HI-Small seulement.")

    sc = HistScorer(a.model)
    bundle = load_bundle("strict")  # noms, percentiles et médianes du train : features communes
    all_names = bundle.forest.engine_features
    median = np.array([bundle.quantiles[all_names[i]][4] for i in sc.cols], dtype=np.float32)
    lo, hi = sc.stream.split_range("val")
    s = 1.0 / (1.0 + np.exp(-sc.logit(np.arange(lo, hi))))
    alerts = lo + np.flatnonzero(s >= sc.threshold)
    faith = {k: faithfulness(sc, alerts, median, sc.threshold, k) for k in (1, 3, 5)}

    names = AccountNames(VARIANT)
    countries = sc.stream.meta["countries"]
    top = int(alerts[np.argmax(s[alerts - lo])])
    example = {"tx_index": top, **reasons(sc, bundle, names, countries, top)}
    keys = ("feature", "text", "contribution", "percentile")
    example["features"] = [{k: f[k] for k in keys} for f in example["features"]]
    write_result("explain_hist", {"model": a.model, "split": "val", "method": "gradient × entrée (features et "
                                  "transactions passées)", "faithfulness": faith, "example": example},
                 seed=42)  # fmt: skip
    print(json.dumps({"faithfulness": faith, "example": example}, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
