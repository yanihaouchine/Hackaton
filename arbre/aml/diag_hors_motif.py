"""Diagnostic : les illicites hors motif (sans typologie dans Patterns.txt) sont-elles séparables des licites
avec les features actuelles ? Lecture seule ; écrit out/diag/diag_hors_motif_<horodatage>.json.

Sous-problème isolé : positifs = illicites hors motif, négatifs = licites ; les illicites de motif sont retirées
partout (entraînement et mesure), pour ne mesurer que cette famille. Test jamais lu.
  - entraînement : train (01–06/09), toutes les illicites hors motif + une fraction des licites ;
  - arrêt précoce : 1er jour de validation (07/09) ;
  - mesure : 2e jour de validation (08/09), jamais vu par l'arrêt précoce.
Scores comparés sur le jour de mesure :
  hasard           : PR-AUC = part des positifs ;
  signaux_faibles  : règle à trois signaux (nouveau couple, banques différentes, ACH), sans apprentissage ;
  strict_actuel    : le modèle livré, tel quel ;
  lgbm_intrinseque : forêt dédiée, features de la transaction et KYC seulement ;
  lgbm_complet     : forêt dédiée, toutes les features du modèle strict.
Lecture : si lgbm_complet reste proche du hasard, ce rappel est hors d'atteinte avec ces features.
Les étiquettes ne servent qu'à ce diagnostic, jamais comme feature.

Lancement (depuis arbre/) : ../.venv/bin/python -m aml.diag_hors_motif [--neg-rate 0.25] [--seeds 42 43 44]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from aml.config import ROOT, SEED
from aml.data import load, load_labels
from aml.evaluate import _curve, best_f1_threshold, pr_auc
from aml.features import load_features
from aml.forest import c_raw
from aml.model import load_bundle
from aml.provenance import run_info
from aml.train import ALL_FAMILIES, DEFAULT_PARAMS, select

DAY = 1440


def _metrics(y: np.ndarray, s: np.ndarray, precisions=(0.5, 0.2, 0.1)) -> dict:
    tps, fps, _ = _curve(y, s)
    p = tps / np.maximum(tps + fps, 1)
    r = tps / max(tps[-1], 1)
    prev = float(np.mean(y))
    ap = pr_auc(y, s)
    out = {"pr_auc": ap, "gain_sur_hasard": ap / prev if prev else None, "meilleur_f1": best_f1_threshold(y, s)}
    for q in precisions:  # rappel maximal en gardant au moins cette précision
        ok = p >= q
        out[f"rappel_a_precision_{q}"] = float(r[ok].max()) if ok.any() else 0.0
    return out


def _fit(X, y, cols, names, tr, es, seed):
    import lightgbm as lgb

    p = dict(DEFAULT_PARAMS, seed=seed)
    fn = [names[i] for i in cols]
    dtr = lgb.Dataset(np.asarray(X[tr][:, cols], dtype=np.float32), y[tr], feature_name=fn, free_raw_data=True)
    des = lgb.Dataset(np.asarray(X[es][:, cols], dtype=np.float32), y[es], reference=dtr, free_raw_data=True)
    b = lgb.train(p, dtr, num_boost_round=3000, valid_sets=[des], callbacks=[lgb.early_stopping(100, verbose=False)])
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="HI-Small")
    ap.add_argument("--neg-rate", type=float, default=0.25, help="fraction des licites du train gardée")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED, SEED + 1, SEED + 2])
    a = ap.parse_args()
    t0 = time.perf_counter()

    s = load(a.variant)
    lab = load_labels(a.variant)
    X, names, _ = load_features(a.variant)
    y_all = lab["y"].astype(np.int8)
    pos = (y_all == 1) & (lab["attempt"] < 0)  # illicites hors motif
    keep = (y_all == 0) | pos  # illicites de motif retirées
    y = pos.astype(np.int8)

    t = np.asarray(s.t)
    tr0, tr1 = s.split_range("train")
    v0, v1 = s.split_range("val")
    vm = int(np.searchsorted(t, t[v0] + DAY - (t[v0] % DAY)))  # minuit entre les deux jours de validation
    rng = np.random.default_rng(SEED)
    r_tr = np.arange(tr0, tr1)
    tr = r_tr[keep[tr0:tr1] & (pos[tr0:tr1] | (rng.random(tr1 - tr0) < a.neg_rate))]
    es = np.arange(v0, vm)[keep[v0:vm]]
    ev = np.arange(vm, v1)[keep[vm:v1]]
    ye = y[ev]
    counts = {
        "train": {"positifs": int(y[tr].sum()), "lignes": len(tr)},
        "arret_precoce_07_09": {"positifs": int(y[es].sum()), "lignes": len(es)},
        "mesure_08_09": {"positifs": int(ye.sum()), "lignes": len(ev)},
    }
    print(json.dumps(counts, ensure_ascii=False), flush=True)

    xe = np.asarray(X[ev], dtype=np.float32)
    col = {n: i for i, n in enumerate(names)}
    rule = (xe[:, col["p_cnt"]] == 0).astype(np.float64) + (xe[:, col["tx_same_bank"]] == 0) + xe[:, col["tx_is_ach"]]
    res = {"hasard": {"pr_auc": float(ye.mean())}, "signaux_faibles": _metrics(ye, rule)}

    b = load_bundle("strict")
    res["strict_actuel"] = _metrics(ye, c_raw(b.forest, xe[:, b.cols]))
    print("strict_actuel", round(res["strict_actuel"]["pr_auc"], 4), flush=True)

    sets = {"lgbm_intrinseque": select(names, ["tx", "kyc"]), "lgbm_complet": select(names, ALL_FAMILIES)}
    for name, cols in sets.items():
        runs, imp = [], np.zeros(len(cols))
        for seed in a.seeds:
            bst = _fit(X, y, cols, names, tr, es, seed)
            m = _metrics(ye, bst.predict(xe[:, cols], num_iteration=bst.best_iteration))
            m["seed"], m["iterations"] = seed, int(bst.best_iteration)
            runs.append(m)
            g = bst.feature_importance("gain", iteration=bst.best_iteration)
            imp += g / max(g.sum(), 1e-12)
            print(f"{name} seed {seed} : PR-AUC {m['pr_auc']:.4f}, F1 {m['meilleur_f1']['f1']:.3f}", flush=True)
        aps = [r["pr_auc"] for r in runs]
        top = np.argsort(-imp)[:15]
        res[name] = {
            "pr_auc_moyenne": float(np.mean(aps)),
            "pr_auc_ecart_type": float(np.std(aps)),
            "par_seed": runs,
            "importance_gain_top15": {names[cols[i]]: round(float(imp[i] / len(a.seeds)), 4) for i in top},
        }

    payload = {
        "run": run_info(SEED),
        "variant": a.variant,
        "args": vars(a),
        "effectifs": counts,
        "seconds": round(time.perf_counter() - t0, 1),
        "resultats": res,
    }
    out = ROOT / "out" / "diag"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"diag_hors_motif_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=lambda o: o.item()) + "\n")
    print(f"\n{path}")


if __name__ == "__main__":
    main()
