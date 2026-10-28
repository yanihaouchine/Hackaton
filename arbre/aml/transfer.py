"""Transfert : un modèle figé (entraîné sur HI-Small) évalué sur une autre variante (HI-Medium, LI-Small…).

La forêt et son seuil de validation sont appliqués tels quels : rien n'est appris ni réglé sur la variante,
qui est donc évaluée en entier. La traîne (fins de tentatives, très majoritairement illicite) est ventilée à part :
« sans traîne » est la ligne de référence, « complet » la gonfle. Les premiers jours partent d'un état vide
(démarrage à froid, comme un déploiement réel) ; ils restent dans le calcul.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from aml.cases import build_cases
from aml.config import DEFAULT_VARIANT, RESULTS
from aml.data import load, load_labels
from aml.evaluate import case_report, evaluate
from aml.features import load_features
from aml.forest import c_raw
from aml.model import load_bundle, load_cascade
from aml.provenance import write_result

CHUNK = 1 << 18  # lignes par bloc : ≈ 0,1 Go de features par thread, quelle que soit la variante
TAIL_FRACTION = 0.01  # la traîne commence au premier jour sous 1 % du volume journalier médian
THREADS = min(8, os.cpu_count() or 1)  # fixé et consigné ; les scores n'en dépendent pas (lignes indépendantes)


def tail_start_day(t: np.ndarray) -> int | None:
    """Premier jour (depuis t0) dont le volume tombe sous TAIL_FRACTION du volume médian ; détecté sur les
    volumes seuls (aucune étiquette)."""
    counts = np.bincount(np.asarray(t) // 1440)
    low = np.flatnonzero(counts < TAIL_FRACTION * np.median(counts[counts > 0]))
    return int(low[0]) if len(low) else None


def score(forest, X, lo: int, hi: int, cascade=None, threads: int = THREADS) -> tuple[np.ndarray, int]:
    """Probabilités sur X[lo:hi], par blocs notés en parallèle (l'appel C relâche le GIL).

    Avec cascade, comme en ligne : le mini-modèle note tout, la forêt complète seulement les lignes au-dessus
    de sa coupure ; les autres gardent le score du mini-modèle. Renvoie aussi le nombre de lignes passées à
    la forêt complète."""
    p = np.empty(hi - lo, dtype=np.float64)

    def block(a: int) -> int:
        b = min(a + CHUNK, hi)
        xs = np.ascontiguousarray(X[a:b], dtype=np.float32)
        if cascade is None:
            r, n_full = c_raw(forest, xs), b - a
        else:
            r = c_raw(cascade[0], xs)
            keep = np.flatnonzero(r >= cascade[1])
            if len(keep):
                r[keep] = c_raw(forest, xs[keep])
            n_full = len(keep)
        p[a - lo : b - lo] = 1.0 / (1.0 + np.exp(-r))
        return n_full

    with ThreadPoolExecutor(max_workers=threads) as ex:
        n_full = sum(ex.map(block, range(lo, hi, CHUNK)))
    return p, n_full


def _reference(model: str) -> dict | None:
    """Derniers chiffres du même modèle sur le test de HI-Small (results/eval_<modèle>_test-…), pour comparaison."""
    files = sorted(RESULTS.glob(f"eval_{model}_test-*.json"))
    if not files:
        return None
    d = json.loads(files[-1].read_text())
    if d.get("variant", DEFAULT_VARIANT) != "HI-Small":  # les premiers fichiers n'enregistraient pas la variante
        return None
    keep = ("precision", "recall", "f1", "pr_auc", "n_alerts")
    return {"file": files[-1].name,
            "splits": {k: {m: v.get(m) for m in keep} for k, v in d["splits"].items()}}  # fmt: skip


def run(variant: str, name: str = "strict", cascade: bool = True, threads: int = THREADS) -> dict:
    b = load_bundle(name)
    if b.info.get("mode") == "j1" or b.info.get("stage2"):
        raise NotImplementedError("transfert prévu pour les modèles stricts (ni J+1, ni étape 2)")
    casc = load_cascade(b) if cascade else None
    if cascade and casc is None:
        raise FileNotFoundError("aucune cascade entraînée pour ce modèle (python -m aml cascade)")
    s = load(variant)
    X, names, fmeta = load_features(variant)
    if names != b.forest.engine_features:
        raise ValueError("le modèle ne correspond pas aux features calculées")
    lab = load_labels(variant)
    y, att, typ = lab["y"], lab["attempt"], lab["typ"]
    t = np.asarray(s.t)

    t_score = time.perf_counter()
    p, n_full = score(b.forest, X, 0, s.n, casc, threads)
    t_score = time.perf_counter() - t_score
    print(
        f"notation : {s.n} tx en {t_score:.1f} s ({s.n / t_score:,.0f} tx/s, {threads} threads), "
        f"forêt complète sur {n_full} tx ({n_full / s.n:.2%})",
        flush=True,
    )

    # début réel de chaque tentative : le délai de détection en part
    m = att >= 0
    order = np.argsort(att[m], kind="stable")
    a_sorted, t_sorted = att[m][order], t[m][order]
    ids, first = np.unique(a_sorted, return_index=True)
    starts = dict(zip(ids.tolist(), np.minimum.reduceat(t_sorted, first).tolist(), strict=True)) if len(ids) else {}

    tail_day = tail_start_day(t)
    cut = s.n if tail_day is None else int(np.searchsorted(t, tail_day * 1440, side="left"))
    windows = {"no_tail": (0, cut), "full": (0, s.n)}
    if cut < s.n:
        windows["tail"] = (cut, s.n)

    out = {}
    for w, (lo, hi) in windows.items():
        sl = slice(lo, hi)
        res = evaluate(y[sl], p[sl], b.threshold, t[sl], att[sl], typ[sl], attempt_start=starts)
        al = np.flatnonzero(p[sl] >= b.threshold)
        ia = lo + al
        cases = build_cases(ia, np.asarray(s.src[ia]), np.asarray(s.dst[ia]), t[ia],
                            np.asarray(s.usd_paid[ia], dtype=np.float64), b.calibrate(p[ia]))  # fmt: skip
        res["cases"] = case_report(cases, lambda idx: y[np.asarray(idx)] == 1)
        res["window_days"] = [int(t[lo] // 1440), int(t[hi - 1] // 1440)] if hi > lo else None
        out[w] = res
        print(f"{name} → {variant} [{w}] P {res['precision']:.4f} R {res['recall']:.4f} F1 {res['f1']:.4f} "
              f"IC95 {res['f1_ci95'][0]:.3f}-{res['f1_ci95'][1]:.3f} PR-AUC {res['pr_auc']:.4f} "
              f"alertes {res['n_alerts']} illicites {res['n_pos']}/{res['n_tx']}", flush=True)  # fmt: skip

    payload = {
        "model": name,
        "trained_on": b.info.get("variant"),
        "variant": variant,
        "threshold": b.threshold,
        "threshold_source": f"validation de {b.info.get('variant')} (figé, non réglé sur {variant})",
        "cascade": cascade,
        "features": fmeta["config_digest"],
        "scoring": {
            "seconds": round(t_score, 2),
            "tx_per_s": round(s.n / t_score),
            "threads": threads,
            "full_forest_rows": n_full,
            "full_forest_share": n_full / s.n,
            "note": "notation par lots des features précalculées ; latence en flux : python -m aml bench",
        },
        "tail_start_day": tail_day,
        "tail_rule": f"premier jour < {TAIL_FRACTION:.0%} du volume journalier médian (volumes seuls)",
        "model_info": {
            k: b.info[k] for k in ("best_iteration", "neg_rate", "families", "forest_digest") if k in b.info
        },  # fmt: skip
        "reference_hi_small": _reference(name + ("_cascade" if cascade else "")),
        "windows": out,
    }
    path = write_result(f"transfer_{name}{'_cascade' if cascade else ''}_{variant}", payload, unique=True)
    print(f"→ {path}", flush=True)
    return payload
