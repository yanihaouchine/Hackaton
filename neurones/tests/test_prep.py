import numpy as np
import pytest
import torch

from mlp.prep import CATEGORICAL, CLIP, HOUR, NO_MATCH, fit_prep

# Colonnes minimales : toutes celles que le prétraitement nomme, plus trois numériques typiques.
NAMES = [*CATEGORICAL, HOUR, *NO_MATCH, "s_age", "tx_log_usd", "s_n_out"]
COL = {n: i for i, n in enumerate(NAMES)}


def sample(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, len(NAMES)), dtype=np.float32)
    for c in CATEGORICAL:
        x[:, COL[c]] = rng.integers(0, 4, n)
    x[:, COL[HOUR]] = rng.integers(0, 24, n)
    for c in NO_MATCH:
        x[:, COL[c]] = rng.choice([0.5, 9.0], n)
    x[:, COL["s_age"]] = rng.choice([-1.0, 100.0, 5000.0], n)
    x[:, COL["tx_log_usd"]] = rng.normal(7, 2, n)
    x[:, COL["s_n_out"]] = rng.exponential(50, n)
    return x


def run(prep, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return prep(torch.from_numpy(x)).numpy()


def test_output_width_matches_out_dim() -> None:
    p = fit_prep(sample(), NAMES)
    assert run(p, sample(10, 1)).shape == (10, p.out_dim)


def test_missing_sentinel_becomes_zero_plus_indicator() -> None:
    p = fit_prep(sample(), NAMES)
    x = sample(2, 1)
    x[0, COL["s_age"]], x[1, COL["s_age"]] = -1.0, 100.0
    d = p.dense(torch.from_numpy(x)).numpy()
    j = p.num_names.index("s_age")
    assert d[0, j] == 0.0 and d[1, j] == pytest.approx(np.log1p(100.0))  # log1p après remplacement
    k = len(p.num_names) + p.miss_idx.tolist().index(j)
    assert d[0, k] == 1.0 and d[1, k] == 0.0


def test_no_match_indicator() -> None:
    p = fit_prep(sample(), NAMES)
    x = sample(2, 1)
    x[0, COL["f_best_err"]], x[1, COL["f_best_err"]] = 9.0, 0.5
    d = p.dense(torch.from_numpy(x)).numpy()
    k = len(p.num_names) + len(p.miss_idx) + NO_MATCH.index("f_best_err")
    assert d[0, k] == 1.0 and d[1, k] == 0.0


def test_unknown_category_gives_zero_vector() -> None:
    p = fit_prep(sample(), NAMES)  # codes 0 à 3 vus sur le train
    x = sample(1, 1)
    x[0, COL["tx_fmt"]] = 50
    start = int(p.mean.numel())
    assert run(p, x)[0, start : start + p.cat_sizes["tx_fmt"]].sum() == 0.0


def test_statistics_depend_only_on_train() -> None:
    train = sample()
    p1 = fit_prep(train, NAMES)
    p2 = fit_prep(train, NAMES)
    other = sample(50, 7) * 3  # données « futures » très différentes
    run(p1, other)
    assert torch.equal(p1.mean, p2.mean) and torch.equal(p1.std, p2.std)


def test_standardized_part_is_centered_on_train() -> None:
    train = sample(5000)
    p = fit_prep(train, NAMES)
    z = run(p, train)[:, : p.mean.numel()]
    assert np.allclose(z.mean(0), 0.0, atol=0.05)


@pytest.mark.parametrize("v", [1e9, -1e9])
def test_extreme_values_are_clipped(v: float) -> None:
    p = fit_prep(sample(), NAMES)
    x = sample(1, 1)
    x[0, COL["tx_log_usd"]] = v
    z = run(p, x)[0, : p.mean.numel()]
    assert np.all(np.abs(z) <= CLIP)


def test_hour_is_cyclic() -> None:
    p = fit_prep(sample(), NAMES)
    x = sample(2, 1)
    x[0, COL[HOUR]], x[1, COL[HOUR]] = 0, 24
    out = run(p, x)
    assert np.allclose(out[0, -2:], out[1, -2:], atol=1e-6)
