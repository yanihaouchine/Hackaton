import numpy as np
import torch
from test_prep import NAMES, sample

from mlp.explain import contributions, owners, to_engine_space
from mlp.prep import fit_prep
from mlp.train import Net, Scorer


def scorer(hidden=(8,)) -> Scorer:
    torch.manual_seed(0)
    return Scorer(Net(fit_prep(sample(), NAMES), hidden, 0.0), 1.0).eval()


def test_one_owner_per_network_input() -> None:
    s = scorer()
    own = owners(s)
    assert len(own) == s.net.prep.out_dim
    assert set(own.tolist()) == set(range(len(NAMES)))  # chaque feature brute alimente au moins une entrée


def test_contributions_shape_and_finite() -> None:
    c = contributions(scorer(), sample(5, 1))
    assert c.shape == (5, len(NAMES)) and np.isfinite(c).all()


def test_linear_net_contribution_is_weight_times_input() -> None:
    """Pour un réseau linéaire, gradient × entrée = somme des poids × entrées prétraitées d'une même feature."""
    s = scorer(hidden=())
    w = s.net.mlp[0].weight.detach().numpy()[0]
    x = sample(1, 1)
    with torch.no_grad():
        z = s.net.prep(torch.from_numpy(x)).numpy()[0]
    expected = np.zeros(len(NAMES))
    np.add.at(expected, owners(s), w * z)
    assert np.allclose(contributions(s, x)[0], expected, atol=1e-5)


def test_engine_space_layout() -> None:
    out = to_engine_space(np.array([[1.0, 2.0]]), cols=[0, 3], n_engine=4)
    assert out.tolist() == [[1.0, 0.0, 0.0, 2.0, 0.0]]
