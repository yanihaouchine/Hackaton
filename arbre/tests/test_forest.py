"""Forêt aplatie : mêmes scores que LightGBM (C et numpy), aller-retour disque, scoring en ligne."""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(0)
    n, d = 20000, 12
    x = rng.normal(size=(n, d)).astype(np.float32)
    x[:, 3] = np.round(x[:, 3])  # valeurs répétées : seuils sur des points exacts
    y = ((x[:, 0] + x[:, 1] * x[:, 2] + 0.5 * (x[:, 3] > 0)) > 0.8).astype(int)
    b = lgb.train({"objective": "binary", "num_leaves": 15, "verbose": -1, "seed": 1, "num_threads": 2},
                  lgb.Dataset(x, y), num_boost_round=40)  # fmt: skip
    return x, b


def test_flat_forest_matches_lightgbm(model, tmp_path):
    from aml.forest import Forest, c_raw, from_lightgbm

    x, b = model
    names = [f"f{i}" for i in range(14)]
    cols = list(range(2, 14))  # le modèle voit les colonnes 2..13 du vecteur moteur
    f = from_lightgbm(b, cols, names)
    xe = np.zeros((len(x), 14), dtype=np.float32)
    xe[:, 2:] = x
    want = b.predict(x, raw_score=True)
    assert np.allclose(f.raw(xe), want, atol=1e-9)
    assert np.allclose(c_raw(f, xe), want, atol=1e-9)
    size = f.save(tmp_path / "m.amlz")
    g = Forest.load(tmp_path / "m.amlz")
    assert size < 200_000
    assert np.allclose(c_raw(g, xe), want, atol=1e-9)  # seuils float32 arrondis vers le bas : décisions exactes
