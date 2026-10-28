"""Qui sont les fraudes ratées par les deux modèles ? Diagnostic sur la VALIDATION uniquement (test non lu).

Fraudes ratées = illicites sous le seuil du MLP livré ET sous celui de la forêt strict ; comparées aux fraudes
attrapées par au moins un des deux. Questions : hors motifs ou non, typologie, format, montant, place dans la
tentative, et surtout historique des comptes (un modèle qui lit l'historique brut aurait-il de quoi apprendre ?).
Usage (depuis neurones/) : python -m mlp.missed  ->  results/val_missed.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort

from mlp.common import ARTIFACTS, load_features, load_stream, write_result


def share(mask: np.ndarray, sub: np.ndarray) -> float:
    return float(mask[sub].mean()) if sub.any() else float("nan")


def account_past(X, cols, stream, lab, sess, thr_mlp, bundle, lo, hi, att, missed, caught, y) -> dict:
    """Part des transactions dont un des deux comptes était déjà apparu (strictement avant, depuis le 01/09) :
    - dans une transaction illicite : diagnostic seulement (étiquettes, jamais utilisable en flux) ;
    - dans une transaction en alerte (MLP ou forêt) : information causale, utilisable en flux."""
    from aml.forest import c_raw

    src, dst, y_all = np.asarray(stream.src[:hi]), np.asarray(stream.dst[:hi]), lab["y"][:hi].astype(bool)
    alert = np.zeros(hi, bool)
    for i in range(0, hi, 262144):
        xa = np.asarray(X[i : min(i + 262144, hi)], dtype=np.float32)
        sm = sess.run(None, {"features": np.ascontiguousarray(xa[:, cols])})[0]
        alert[i : i + len(xa)] = (sm >= thr_mlp) | (1.0 / (1.0 + np.exp(-c_raw(bundle.forest, xa))) >= bundle.threshold)
    n_acc = int(max(src.max(), dst.max())) + 1
    ill_seen, al_seen = np.zeros(n_acc, bool), np.zeros(n_acc, bool)
    prior_ill, prior_al = np.zeros(hi, bool), np.zeros(hi, bool)
    for j in range(hi):  # parcours chronologique : marquage après la transaction, donc strictement antérieur
        s_, d_ = src[j], dst[j]
        prior_ill[j] = ill_seen[s_] or ill_seen[d_]
        prior_al[j] = al_seen[s_] or al_seen[d_]
        if y_all[j]:
            ill_seen[s_] = ill_seen[d_] = True
        if alert[j]:
            al_seen[s_] = al_seen[d_] = True
    pi, pa = prior_ill[lo:hi], prior_al[lo:hi]
    groups = {"missed": missed, "missed_non_pattern": missed & (att < 0), "caught": caught, "licit": ~y}
    return {g: {"n": int(m.sum()), "account_in_earlier_illicit": share(pi, m), "account_in_earlier_alert": share(pa, m)}
            for g, m in groups.items()}  # fmt: skip


def main() -> None:
    from aml.config import FORMATS, TYPOLOGIES
    from aml.forest import c_raw
    from aml.model import load_bundle

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    a = p.parse_args()

    X, names, cols = load_features()
    stream, lab = load_stream()
    lo, hi = stream.split_range("val")
    x_all = np.asarray(X[lo:hi], dtype=np.float32)
    f = {n: x_all[:, i] for i, n in enumerate(names)}
    y = lab["y"][lo:hi].astype(bool)
    att, typ = lab["attempt"][lo:hi], lab["typ"][lo:hi]

    meta = json.loads((ARTIFACTS / a.model / "meta.json").read_text())
    sess = ort.InferenceSession(str(ARTIFACTS / a.model / "model.onnx"), providers=["CPUExecutionProvider"])
    x = np.ascontiguousarray(x_all[:, cols])
    s_mlp = np.concatenate([sess.run(None, {"features": x[i : i + 65536]})[0] for i in range(0, len(x), 65536)])
    bundle = load_bundle("strict")
    s_for = 1.0 / (1.0 + np.exp(-c_raw(bundle.forest, x_all)))
    caught = y & ((s_mlp >= meta["threshold"]) | (s_for >= bundle.threshold))
    missed = y & ~caught

    # Rang de la transaction dans sa tentative (toutes périodes : le début peut précéder la validation)
    att_all, t_all = lab["attempt"], np.asarray(stream.t)
    pos = np.full(hi - lo, -1)
    for k in np.unique(att[att >= 0]):
        idx_all = np.flatnonzero(att_all == k)
        order = idx_all[np.argsort(t_all[idx_all], kind="stable")]
        rank = {int(j): r for r, j in enumerate(order)}
        for j in np.flatnonzero(att == k):
            pos[j] = rank[int(lo + j)]

    groups = {"missed": missed, "caught": caught}
    out: dict = {"model": a.model, "split": "val", "n": {g: int(m.sum()) for g, m in groups.items()}}
    out["non_pattern_share"] = {g: share(att < 0, m) for g, m in groups.items()}
    out["by_typology_missed_over_total"] = {
        name: f"{int(np.sum(missed & (typ == k)))}/{int(np.sum(y & (typ == k)))}" for k, name in enumerate(TYPOLOGIES)
    }
    out["by_format"] = {FORMATS[k]: {g: int(np.sum(m & (f['tx_fmt'] == k))) for g, m in groups.items()}
                        for k in range(len(FORMATS)) if np.any(y & (f["tx_fmt"] == k))}  # fmt: skip
    out["amount_usd_median"] = {g: float(np.median(np.expm1(f["tx_log_usd"][m]))) for g, m in groups.items()}
    out["first_tx_of_attempt_share"] = {g: share(pos == 0, m & (att >= 0)) for g, m in groups.items()}
    # Historique disponible au moment de la transaction (features causales du moteur)
    hist = {
        "sender_prior_tx_median": lambda m: float(np.median(f["s_n_out"][m] + f["s_n_in"][m])),
        "receiver_prior_tx_median": lambda m: float(np.median(f["r_n_out"][m] + f["r_n_in"][m])),
        "sender_no_history_share": lambda m: share((f["s_n_out"] + f["s_n_in"]) == 0, m),
        "receiver_no_history_share": lambda m: share((f["r_n_out"] + f["r_n_in"]) == 0, m),
        "sender_8plus_prior_tx_share": lambda m: share((f["s_n_out"] + f["s_n_in"]) >= 8, m),
        "receiver_8plus_prior_tx_share": lambda m: share((f["r_n_out"] + f["r_n_in"]) >= 8, m),
        "chain_depth_positive_share": lambda m: share(f["f_chain_depth"] > 0, m),
    }
    out["history"] = {k: {g: fn(m) for g, m in groups.items()} for k, fn in hist.items()}
    out["score_median"] = {g: {"mlp": float(np.median(s_mlp[m])), "forest": float(np.median(s_for[m]))}
                           for g, m in groups.items()}  # fmt: skip
    out["thresholds"] = {"mlp": meta["threshold"], "forest": bundle.threshold}
    out["account_past"] = account_past(X, cols, stream, lab, sess, meta["threshold"], bundle, lo, hi, att, missed,
                                       caught, y)  # fmt: skip
    write_result("val_missed", out, seed=42)
    print(json.dumps({k: v for k, v in out.items() if k not in ("model", "split")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
