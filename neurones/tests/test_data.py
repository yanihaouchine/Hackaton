import numpy as np
import pytest

from nn.data import Preprocessing, fit_preprocessing, signed_log1p

NAMES = ["tx_log_usd", "tx_fmt", "s_age"]  # numérique, catégorie one-hot, numérique avec sentinelle -1


def toy() -> np.ndarray:
    return np.array([[1.0, 0, -1], [2.0, 2, 10], [3.0, 1, 100]], dtype=np.float32)


def test_signed_log1p_keeps_sentinel_apart() -> None:
    never, zero = signed_log1p(np.array([-1.0, 0.0]))
    assert never < 0 and zero == 0


def test_one_hot_and_shape() -> None:
    pre = fit_preprocessing(toy(), NAMES)
    assert pre.cat_sizes == [3] and pre.n_inputs == 2 + 3
    z = pre.transform(np.array([[1.0, 2, 0]], dtype=np.float32))
    assert z[0, 2:].tolist() == [0.0, 0.0, 1.0]


def test_unknown_category_gives_zero_vector() -> None:
    pre = fit_preprocessing(toy(), NAMES)
    z = pre.transform(np.array([[1.0, 6, 0]], dtype=np.float32))  # code 6 jamais vu sur le train
    assert z[0, 2:].tolist() == [0.0, 0.0, 0.0]


def test_numeric_centered_reduced_on_train() -> None:
    pre = fit_preprocessing(toy(), NAMES)
    z = pre.transform(toy())[:, :2]
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-4)


def test_statistics_depend_only_on_train(tmp_path) -> None:
    """Transformer d'autres données (validation, test) ne modifie pas les paramètres appris."""
    pre = fit_preprocessing(toy(), NAMES)
    before = pre.num_mean.copy()
    pre.transform(np.full((4, 3), 1e6, dtype=np.float32))
    assert pre.num_mean == before
    pre.save(tmp_path / "p.json")
    assert Preprocessing.load(tmp_path / "p.json") == pre


def test_constant_column_does_not_divide_by_zero() -> None:
    x = toy()
    x[:, 0] = 5.0
    z = fit_preprocessing(x, NAMES).transform(x)
    assert np.isfinite(z).all()


@pytest.mark.parametrize("v", [1e30, -1e30])
def test_extreme_values_are_clipped(v: float) -> None:
    pre = fit_preprocessing(toy(), NAMES)
    z = pre.transform(np.array([[v, 0, 0]], dtype=np.float32))
    assert abs(z[0, 0]) <= 10.0
