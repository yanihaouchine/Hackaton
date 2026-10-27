import numpy as np
import torch

from nn.data import fit_preprocessing
from nn.model import Normalize, Scorer, mlp

NAMES = ["tx_log_usd", "tx_fmt", "s_age", "kyc_s_country"]


def data(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.exponential(5, n), rng.integers(0, 7, n), rng.integers(-1, 6000, n), rng.integers(0, 34, n)]
    ).astype(np.float32)


def test_torch_normalization_matches_numpy() -> None:
    """La normalisation exportée en ONNX (torch) est la même que celle de l'entraînement (numpy)."""
    pre = fit_preprocessing(data(), NAMES)
    x = data(seed=1)
    x[0, 1] = 9  # catégorie jamais vue
    got = Normalize(pre)(torch.from_numpy(x)).numpy()
    assert got.shape == (len(x), pre.n_inputs)
    assert np.allclose(got, pre.transform(x), atol=1e-5)


def test_scorer_outputs_probabilities() -> None:
    pre = fit_preprocessing(data(), NAMES)
    torch.manual_seed(0)
    p = Scorer(pre, mlp(pre.n_inputs)).eval()(torch.from_numpy(data(seed=2))).detach().numpy()
    assert p.shape == (200,) and ((p > 0) & (p < 1)).all()
