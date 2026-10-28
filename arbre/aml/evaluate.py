"""Métriques sur la classe illicite : P/R/F1 au seuil, PR-AUC, précision@k par jour, rappel par typologie.

Toutes les fonctions sont en numpy pur (pas de dépendance d'évaluation) et testées sur des cas jouets.
"""

from __future__ import annotations

import numpy as np

from aml.config import TYPOLOGIES


def _curve(y: np.ndarray, s: np.ndarray):
    """Points (vrais positifs, faux positifs, seuil) pour chaque seuil distinct, scores décroissants."""
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]  # dernier indice de chaque groupe ex aequo
    tps = np.cumsum(ys)[last]
    fps = (last + 1) - tps
    return tps, fps, ss[last]


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Précision moyenne (définition en escalier, identique à sklearn.average_precision_score)."""
    tps, fps, _ = _curve(y, s)
    if tps[-1] == 0:
        return float("nan")
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> dict:
    """Seuil maximisant le F1 (alerte si score >= seuil)."""
    tps, fps, thr = _curve(y, s)
    p = tps / np.maximum(tps + fps, 1)
    r = tps / max(tps[-1], 1)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return {"threshold": float(thr[i]), "f1": float(f1[i]), "precision": float(p[i]), "recall": float(r[i])}


def binary_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    y = np.asarray(y).astype(bool)
    a = np.asarray(s) >= thr
    tp = int(np.sum(a & y))
    fp = int(np.sum(a & ~y))
    fn = int(np.sum(~a & y))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "n_alerts": tp + fp, "precision": p, "recall": r, "f1": f1}


def bootstrap_f1(tp: int, fp: int, fn: int, n: int, b: int = 2000, seed: int = 42) -> list[float]:
    """IC 95 % du F1 par bootstrap i.i.d. des transactions (tirage multinomial des 4 cases).

    Limite : ignore la dépendance entre transactions d'une même tentative (IC optimiste).
    """
    rng = np.random.default_rng(seed)
    tn = n - tp - fp - fn
    draws = rng.multinomial(n, np.array([tp, fp, fn, tn]) / n, size=b)
    t, f, m = draws[:, 0], draws[:, 1], draws[:, 2]
    f1 = np.where(2 * t + f + m > 0, 2 * t / np.maximum(2 * t + f + m, 1), 0.0)
    return [float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))]


def precision_at_k_daily(y: np.ndarray, s: np.ndarray, t: np.ndarray, k: int) -> dict:
    """Précision et rappel des k meilleurs scores de chaque jour (capacité quotidienne d'un analyste)."""
    y = np.asarray(y).astype(bool)
    day = np.asarray(t) // 1440
    precs, hits, pos = [], 0, 0
    for d in np.unique(day):
        m = np.flatnonzero(day == d)
        top = m[np.argsort(-np.asarray(s)[m], kind="stable")[:k]]
        precs.append(float(y[top].mean()) if len(top) else 0.0)
        hits += int(y[top].sum())
        pos += int(y[m].sum())
    return {"k": k, "precision_mean": float(np.mean(precs)), "recall": hits / pos if pos else 0.0,
            "n_days": len(precs)}  # fmt: skip


def typology_report(y: np.ndarray, alert: np.ndarray, attempt: np.ndarray, typ: np.ndarray, t: np.ndarray,
                    decision_t: np.ndarray | None = None, attempt_start: dict | None = None) -> dict:  # fmt: skip
    """Par typologie : tentatives détectées (au moins une alerte), rappel en transactions, délai de détection.

    Les tentatives sont restreintes à leurs transactions présentes dans la période évaluée. Le délai va du
    début de la tentative à l'instant de décision de la première alerte (= t en temps réel, minuit en J+1).
    """
    y = np.asarray(y).astype(bool)
    alert = np.asarray(alert).astype(bool)
    dec = np.asarray(t) if decision_t is None else np.asarray(decision_t)
    # indices de chaque tentative, regroupés une fois (un balayage par tentative est trop lent sur HI-Medium)
    pat_idx = np.flatnonzero(attempt >= 0)
    order = pat_idx[np.argsort(attempt[pat_idx], kind="stable")]
    a_ids, first = np.unique(attempt[order], return_index=True)
    groups = dict(zip(a_ids.tolist(), np.split(order, first[1:]), strict=True))
    out = {}
    for k, name in enumerate(TYPOLOGIES):
        m = typ == k
        ids = np.unique(attempt[m])
        det, delays, before_end = 0, [], 0
        for a in ids:
            idx = groups.get(int(a))
            if idx is None:
                idx = np.flatnonzero(attempt == a)
            al = idx[alert[idx]]
            if len(al):
                det += 1
                start = attempt_start.get(int(a), t[idx].min()) if attempt_start else t[idx].min()
                delays.append((dec[al].min() - start) / 60.0)  # depuis le vrai début de la tentative
                before_end += int(dec[al].min() < t[idx].max())
        out[name] = {
            "attempts": int(len(ids)),
            "attempts_detected": det,
            "attempt_recall": det / len(ids) if len(ids) else None,
            "tx": int(m.sum()),
            "tx_recall": float(alert[m].mean()) if m.any() else None,
            "median_delay_h": float(np.median(delays)) if delays else None,
            "detected_before_last_tx": before_end,
        }
    pat = attempt >= 0
    out["_all_patterns"] = {"tx": int(pat.sum()), "tx_recall": float(alert[pat].mean()) if pat.any() else None}
    non = y & (attempt < 0)
    out["_non_pattern"] = {"tx": int(non.sum()), "tx_recall": float(alert[non].mean()) if non.any() else None}
    return out


def evaluate(y, s, thr, t, attempt, typ, ks=(10, 25, 50, 100), decision_t=None, attempt_start=None) -> dict:
    """Bilan complet d'une période : seuil fixé ailleurs (validation), jamais choisi ici."""
    y = np.asarray(y).astype(np.int8)
    s = np.asarray(s)
    res = binary_at(y, s, thr)
    res["threshold"] = float(thr)
    res["n_tx"] = int(len(y))
    res["n_pos"] = int(y.sum())
    res["pr_auc"] = pr_auc(y, s)
    res["f1_ci95"] = bootstrap_f1(res["tp"], res["fp"], res["fn"], len(y))
    res["p_at_k"] = [precision_at_k_daily(y, s, t, k) for k in ks]
    res["typology"] = typology_report(y, s >= thr, attempt, typ, t, decision_t, attempt_start)
    return res


def case_report(cases, y_by_index, ks=(5, 10, 25, 50)) -> dict:
    """Niveau cas (unité de travail de l'analyste) : un cas est « vrai » s'il contient au moins une opération
    illicite. Précision des cas, charge par jour et précision@k des cas classés par priorité chaque jour."""
    if not cases:
        return {"n_cases": 0}
    true = np.array([bool(np.any(y_by_index(c.alerts))) for c in cases])
    day = np.array([c.t_first // 1440 for c in cases])
    prio = np.array([c.priority for c in cases])
    usd = np.array([c.total_usd for c in cases])
    out = {"n_cases": len(cases), "case_precision": float(true.mean()), "cases_per_day": {}, "p_at_k": []}
    for d in np.unique(day):
        out["cases_per_day"][int(d)] = int((day == d).sum())
    for k in ks:
        precs = []
        for d in np.unique(day):
            m = np.flatnonzero(day == d)
            order = m[np.lexsort((-usd[m], -prio[m]))][:k]
            precs.append(float(true[order].mean()))
        out["p_at_k"].append({"k": k, "precision_mean": float(np.mean(precs))})
    return out
