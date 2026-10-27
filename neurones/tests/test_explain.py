import numpy as np
import torch

from nn.data import fit_preprocessing
from nn.explain import contributions, owners, to_engine_space
from nn.model import Scorer, mlp

NAMES = ["tx_log_usd", "tx_fmt", "s_age"]


def scorer() -> Scorer:
    rng = np.random.default_rng(0)
    x = np.column_stack([rng.exponential(5, 100), rng.integers(0, 4, 100), rng.integers(-1, 900, 100)])
    torch.manual_seed(0)
    return Scorer(fit_preprocessing(x.astype(np.float32), NAMES), mlp(2 + 4)).eval()


def test_one_contribution_per_raw_feature() -> None:
    s = scorer()
    assert owners(s).tolist() == [0, 2, 1, 1, 1, 1]  # 2 numériques puis le one-hot de tx_fmt
    c = contributions(s, np.array([[3.0, 2, 10], [1.0, 0, -1]], dtype=np.float32))
    assert c.shape == (2, 3) and np.isfinite(c).all()


def test_linear_net_contribution_is_weight_times_input() -> None:
    """Pour un réseau linéaire, gradient × entrée = poids × entrée normalisée (cas vérifiable à la main)."""
    s = scorer()
    s.net = torch.nn.Sequential(torch.nn.Linear(6, 1, bias=False))
    w = s.net[0].weight.detach().numpy()[0]
    x = np.array([[3.0, 2, 10]], dtype=np.float32)
    z = s.norm(torch.from_numpy(x)).numpy()[0]
    c = contributions(s, x)[0]
    assert np.isclose(c[0], w[0] * z[0]) and np.isclose(c[2], w[1] * z[1])
    assert np.isclose(c[1], w[4] * z[4])  # tx_fmt = 2 → seule la 3e colonne one-hot vaut 1


def test_engine_space_layout() -> None:
    out = to_engine_space(np.array([[1.0, 2.0]]), cols=[0, 3], n_engine=4)
    assert out.tolist() == [[1.0, 0.0, 0.0, 2.0, 0.0]]
