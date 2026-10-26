"""Métriques sur la classe illicite. Les étiquettes ne servent qu'ici, jamais dans la détection."""

from __future__ import annotations

from collections import defaultdict

import polars as pl

from aml.schema import Alert


def prf(tp: int, n_pred: int, n_pos: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_pos if n_pos else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def average_precision(scores: dict[int, float], positives: set[int]) -> float:
    """PR-AUC (précision moyenne) : les transactions sans score sont classées en dernier."""
    if not positives:
        return 0.0
    hits, total = 0, 0.0
    for rank, tx in enumerate(sorted(scores, key=lambda i: (-scores[i], i)), start=1):
        if tx in positives:
            hits += 1
            total += hits / rank
    return total / len(positives)


def evaluate(alerts: list[Alert], period: pl.DataFrame, attempts: pl.DataFrame, k: int) -> dict:
    """`period` : tx_id, ts, is_laundering, payment_format de la période évaluée."""
    in_period = set(period["tx_id"].to_list())
    positives = set(period.filter(pl.col("is_laundering") == 1)["tx_id"].to_list())
    in_patterns = set(attempts["tx_id"].to_list())

    scores: dict[int, float] = {}
    for a in alerts:
        for i in a.tx_ids:
            if i in in_period and scores.get(i, -1.0) < a.score:
                scores[i] = a.score
    flagged = set(scores)
    tx_level = prf(len(flagged & positives), len(flagged), len(positives))
    tx_level["pr_auc"] = average_precision(scores, positives)
    tx_level["n_flagged"] = len(flagged)
    tx_level["n_positives"] = len(positives)
    tx_level["prevalence"] = len(positives) / len(in_period) if in_period else 0.0

    off = positives - in_patterns
    recall_outside = len(flagged & off) / len(off) if off else 0.0

    def is_good(a: Alert) -> bool:
        return a.trigger_tx_id in positives

    per_day: dict[object, list[Alert]] = defaultdict(list)
    for a in alerts:
        per_day[a.triggered_at.date()].append(a)
    p_at_k = []
    for day_alerts in per_day.values():
        top = sorted(day_alerts, key=lambda a: (-a.score, a.alert_id))[:k]
        p_at_k.append(sum(map(is_good, top)) / len(top))

    by_typ: dict[str, dict[str, float]] = {}
    for typ in sorted({a.typology for a in alerts}):
        sub = [a for a in alerts if a.typology == typ]
        by_typ[typ] = {"n_alerts": len(sub), "precision": sum(map(is_good, sub)) / len(sub)}

    att = attempts.filter(pl.col("tx_id").is_in(list(in_period)))
    detected = (
        att.with_columns(pl.col("tx_id").is_in(list(flagged)).alias("hit"))
        .group_by("typology", "attempt_id")
        .agg(pl.col("hit").any())
        .group_by("typology")
        .agg(pl.len().alias("attempts"), pl.col("hit").sum().alias("detected"))
        .sort("typology")
    )
    attempts_by_typ = {
        typ: {"attempts": n, "detected": d, "rate": d / n} for typ, n, d in detected.iter_rows()
    }

    return {
        "n_tx": len(in_period),
        "n_alerts": len(alerts),
        "alerts_per_day": len(alerts) / max(len(per_day), 1),
        "alert_precision": sum(map(is_good, alerts)) / len(alerts) if alerts else 0.0,
        "precision_at_k": {"k": k, "mean_over_days": sum(p_at_k) / len(p_at_k) if p_at_k else 0.0},
        "transactions": tx_level,
        "recall_outside_patterns": recall_outside,
        "alerts_by_typology": by_typ,
        "attempts_detected_by_typology": attempts_by_typ,
    }


def baseline_ach(period: pl.DataFrame) -> dict[str, float]:
    """Baseline naïve : signaler tous les virements ACH (format le plus risqué sur l'entraînement)."""
    pred = period["payment_format"] == "ACH"
    pos = period["is_laundering"] == 1
    return prf(int((pred & pos).sum()), int(pred.sum()), int(pos.sum()))
