"""Métriques : comparaison à sklearn et cas jouets calculables à la main."""

import numpy as np
import pytest

from aml.evaluate import best_f1_threshold, binary_at, pr_auc, precision_at_k_daily, typology_report


def test_pr_auc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    s = rng.random(5000) + y * 0.3
    s[:100] = np.round(s[:100], 1)  # ex aequo
    assert pr_auc(y, s) == pytest.approx(sk.average_precision_score(y, s), abs=1e-12)


def test_best_f1_and_binary():
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    b = best_f1_threshold(y, s)
    assert b["threshold"] == 0.6 and b["f1"] == pytest.approx(2 * 0.75 * 1 / 1.75)
    r = binary_at(y, s, 0.6)
    assert (r["tp"], r["fp"], r["fn"]) == (3, 1, 0)


def test_precision_at_k_daily():
    y = np.array([1, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.9, 0.3])
    t = np.array([0, 10, 20, 1440, 1450, 1460])
    r = precision_at_k_daily(y, s, t, 1)
    assert r["precision_mean"] == 1.0 and r["recall"] == pytest.approx(2 / 3)


def test_typology_report():
    y = np.array([1, 1, 1, 0, 1])
    attempt = np.array([0, 0, 1, -1, -1])
    typ = np.array([0, 0, 2, -1, -1])
    t = np.array([0, 60, 120, 130, 140])
    alert = np.array([False, True, False, True, True])
    r = typology_report(y, alert, attempt, typ, t)
    assert r["FAN-OUT"]["attempts_detected"] == 1 and r["FAN-OUT"]["median_delay_h"] == 1.0
    assert r["CYCLE"]["attempts_detected"] == 0
    assert r["_non_pattern"]["tx_recall"] == 1.0


def test_cases_union_and_report():
    from aml.cases import build_cases
    from aml.evaluate import case_report

    # alertes 0-1 partagent le compte 2 à moins de 96 h ; l'alerte 2 est isolée ; l'alerte 3 réutilise le
    # compte 2 mais 5 jours plus tard (hors fenêtre) : nouveau cas
    idx = np.array([10, 11, 12, 13])
    src = np.array([1, 2, 7, 2])
    dst = np.array([2, 3, 8, 9])
    t = np.array([0, 60, 100, 8000])
    usd = np.array([100.0, 50.0, 10.0, 5.0])
    p = np.array([0.9, 0.5, 0.2, 0.1])
    cases = build_cases(idx, src, dst, t, usd, p)
    groups = sorted(sorted(c.alerts) for c in cases)
    assert groups == [[10, 11], [12], [13]]
    top = cases[0]
    assert top.alerts == [10, 11] and abs(top.priority - (1 - 0.1 * 0.5)) < 1e-12
    y = np.zeros(20, dtype=int)
    y[11] = 1
    r = case_report(cases, lambda a: y[np.asarray(a)] == 1, ks=(1,))
    assert r["n_cases"] == 3 and abs(r["case_precision"] - 1 / 3) < 1e-12
