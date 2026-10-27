"""Moteur C : équivalence avec l'oracle Python, causalité (préfixe, futur perturbé), déterminisme."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from aml.config import DATA_PROC, FORMATS, EngineConfig

pytestmark = pytest.mark.skipif(
    not (DATA_PROC / "HI-Small" / "meta.json").exists(), reason="données préparées absentes (python -m aml prepare)"
)

ACH = FORMATS.index("ACH")


@pytest.fixture(scope="module")
def sample():
    """Flux de test riche en motifs : ACH des jours 2 à 4 + 2 % des autres formats (ordre conservé)."""
    from aml.data import load
    from aml.engine import TX_FIELDS

    s = load("HI-Small")
    lo, hi = np.searchsorted(s.t, 1440), np.searchsorted(s.t, 4 * 1440)
    rng = np.random.default_rng(0)
    fmt = np.asarray(s.fmt[lo:hi])
    idx = lo + np.flatnonzero((fmt == ACH) | (rng.random(hi - lo) < 0.02))
    cols = {k: np.asarray(getattr(s, k)[idx]) for k, _ in TX_FIELDS}
    return s, cols


def _run(s, cols, cfg=None):
    from aml.engine import Engine

    eng = Engine.for_stream(s, cfg)
    x = eng.process(cols)
    st = eng.stats()
    eng.close()
    return x, st


def test_no_label_in_api():
    from aml.engine import TX_FIELDS, Engine

    fields = {k for k, _ in TX_FIELDS}
    assert not fields & {"y", "label", "attempt", "typ", "is_laundering"}
    assert "y" not in inspect.signature(Engine.process).parameters


def test_matches_python_reference(sample):
    from reference_engine import RefEngine

    from aml.engine import feature_names

    s, cols = sample
    n = len(cols["t"])  # ~180 k tx : assez long pour contenir des scatter-gather complets
    sub = cols
    x, stats = _run(s, sub)
    assert stats["scan_cap_hits"] == 0 and stats["ring_overflow"] == 0
    cfg = EngineConfig()
    ref = RefEngine(cfg, cfg.fmt_mask())
    names = feature_names()
    col = {nm: i for i, nm in enumerate(names)}
    rows = []
    for i in range(n):
        rows.append(ref.step(int(sub["t"][i]), int(sub["src"][i]), int(sub["dst"][i]),
                             np.float32(sub["usd_paid"][i]), np.float32(sub["usd_recv"][i]),
                             int(sub["fmt"][i])))  # fmt: skip
    for key in rows[0]:
        want = np.array([r[key] for r in rows], dtype=np.float64)
        got = x[:, col[key]].astype(np.float64)
        bad = ~np.isclose(got, want, rtol=1e-5, atol=1e-5)
        assert not bad.any(), (
            f"{key}: {bad.sum()} écarts, ex. ligne {np.flatnonzero(bad)[0]} C={got[bad][0]} ref={want[bad][0]}"
        )
    # l'échantillon doit vraiment exercer les motifs
    assert (x[:, col["m_cyc_n"]] > 0).sum() > 0
    assert (x[:, col["m_sg"]] > 0).sum() > 0
    assert (x[:, col["f_chain_depth"]] > 0).sum() > 0


def test_prefix_invariance_and_future_perturbation(sample):
    s, cols = sample
    x_full, _ = _run(s, cols)
    n = len(cols["t"]) // 2
    x_pref, _ = _run(s, {k: v[:n] for k, v in cols.items()})
    assert np.array_equal(x_full[:n], x_pref), "les features du passé dépendent du futur"
    rng = np.random.default_rng(1)
    pert = {k: v.copy() for k, v in cols.items()}
    m = len(cols["t"]) - n
    pert["src"][n:] = rng.permutation(pert["src"][n:])
    pert["dst"][n:] = rng.permutation(pert["dst"][n:])
    pert["usd_paid"][n:] = rng.random(m).astype(np.float32) * 1e5
    pert["usd_recv"][n:] = pert["usd_paid"][n:]
    x_pert, _ = _run(s, pert)
    assert np.array_equal(x_full[:n], x_pert[:n])
    assert not np.array_equal(x_full[n:], x_pert[n:])


def test_deterministic(sample):
    s, cols = sample
    a, _ = _run(s, cols)
    b, _ = _run(s, cols)
    assert a.tobytes() == b.tobytes()


def test_propagation_is_causal(sample):
    """Avec une forêt attachée, les scores propagés (z_*) ne dépendent que des transactions passées."""
    lgb = pytest.importorskip("lightgbm")
    from aml.engine import Engine, feature_names
    from aml.forest import from_lightgbm

    s, cols = sample
    names = feature_names()
    x0, _ = _run(s, cols)
    base = [i for i, nm in enumerate(names) if not nm.startswith("z_")]
    yy = (x0[:, names.index("f_chain_depth")] > 0).astype(int)  # cible jouet, sans étiquette réelle
    b = lgb.train({"objective": "binary", "num_leaves": 7, "verbose": -1, "num_threads": 2},
                  lgb.Dataset(x0[:, base], yy), num_boost_round=10)  # fmt: skip
    forest = from_lightgbm(b, base, names)

    def run(c):
        eng = Engine.for_stream(s)
        eng.set_forest(forest)
        sc = np.empty(len(c["t"]), dtype=np.float64)
        x = eng.process(c, scores=sc)
        eng.close()
        return x, sc

    x_full, sc_full = run(cols)
    z = [names.index(nm) for nm in names if nm.startswith("z_")]
    assert np.abs(x_full[:, z]).sum() > 0, "la propagation doit être active"
    n = len(cols["t"]) // 2
    x_pref, sc_pref = run({k: v[:n] for k, v in cols.items()})
    assert np.array_equal(x_full[:n], x_pref) and np.array_equal(sc_full[:n], sc_pref)


def test_snapshot_restore_roundtrip(sample, tmp_path):
    """Un état restauré depuis un instantané donne exactement les mêmes features que l'état d'origine."""
    from aml.engine import Engine

    s, cols = sample
    n = len(cols["t"]) // 2
    first = {k: v[:n] for k, v in cols.items()}
    second = {k: v[n:] for k, v in cols.items()}
    a = Engine.for_stream(s)
    a.process(first)
    size = a.snapshot(tmp_path / "etat.snap")
    want = a.process(second)
    b = Engine.for_stream(s)
    b.restore(tmp_path / "etat.snap")
    got = b.process(second)
    assert size > 0 and np.array_equal(want, got)


def test_hindsight_only_uses_data_before_decision(sample):
    """Revue J+1 : les features rétrospectives d'un jour (décision à minuit) ne dépendent pas des jours suivants."""
    from aml.engine import Engine

    s, cols = sample
    days = cols["t"] // 1440

    def run(c):
        eng = Engine.for_stream(s)
        d = c["t"] // 1440
        bounds = np.r_[0, np.flatnonzero(np.diff(d)) + 1, len(d)]
        out = []
        for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
            part = {k: v[lo:hi] for k, v in c.items()}
            seq = np.empty(hi - lo, dtype=np.int64)
            eng.process(part, seq_at=seq)
            out.append(eng.hindsight(part, seq))
        eng.close()
        return np.concatenate(out)

    full = run(cols)
    n = int(np.searchsorted(days, days[0] + 1, side="left"))  # premier jour seulement
    first = run({k: v[:n] for k, v in cols.items()})
    assert np.array_equal(full[:n], first)
    assert np.abs(full).sum() > 0
