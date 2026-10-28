"""Raisons d'une alerte du MLP : contributions « gradient × entrée », au format des raisons de arbre/.

Méthode reprise de la v1 de neurones/ (commit 0c12204). TreeSHAP ne s'applique qu'aux arbres : pour le réseau,
la contribution d'une entrée prétraitée z est ∂logit/∂z × z, l'effet de l'entrée par rapport à la transaction
moyenne du train (z = 0 après standardisation). Les entrées issues d'une même feature du moteur (one-hot,
indicateurs « absent » et « aucun appariement », sinus et cosinus de l'heure) sont additionnées : une
contribution par feature du moteur. Les phrases françaises et les rangs viennent de `aml.explain.reasons`.
"""

from __future__ import annotations

import numpy as np
import torch

from mlp.train import Scorer


def owners(scorer: Scorer) -> np.ndarray:
    """Pour chaque entrée du réseau, l'indice de la feature brute (parmi les colonnes du modèle) dont elle vient."""
    return prep_owners(scorer.net.prep)


def prep_owners(p) -> np.ndarray:
    """Même chose à partir du prétraitement seul (sert aussi au modèle séquentiel).

    Même ordre que `Prep.forward` : numériques, indicateurs « absent », indicateurs « aucun appariement »,
    one-hot, puis sinus et cosinus de l'heure.
    """
    num = p.num_idx.numpy()
    own = list(num) + list(num[p.miss_idx.numpy()]) + list(num[p.nomatch_idx.numpy()])
    for j, size in p.cat:
        own += [j] * size
    own += [p.hour, p.hour]
    return np.array(own)


def contributions(scorer: Scorer, x: np.ndarray) -> np.ndarray:
    """x : (n, colonnes du modèle) brutes → (n, colonnes du modèle) contributions au logit."""
    net = scorer.net
    z = net.prep(torch.from_numpy(np.asarray(x, dtype=np.float32))).detach().requires_grad_(True)
    net.mlp(z).sum().backward()
    gz = (z.grad * z).detach().numpy()
    out = np.zeros((len(x), x.shape[1]))
    np.add.at(out.T, owners(scorer), gz.T)
    return out


def to_engine_space(contrib: np.ndarray, cols: list[int], n_engine: int) -> np.ndarray:
    """Place les contributions dans l'espace des features du moteur (+ 1 colonne de biais), comme TreeSHAP."""
    out = np.zeros((len(contrib), n_engine + 1))
    out[:, cols] = contrib
    return out


def reasons(scorer: Scorer, bundle, x_engine: np.ndarray, cols: list[int], countries: list[str], k: int = 5):
    """Raisons d'une transaction (vecteur des 104 features du moteur), au format de `aml.explain.reasons`."""
    from aml.explain import reasons as forest_style_reasons

    c = contributions(scorer, x_engine[None, cols])
    row = to_engine_space(c, cols, len(x_engine))[0]
    return forest_style_reasons(bundle, x_engine, row, countries, k=k)


def faithfulness(scorer: Scorer, x: np.ndarray, median: np.ndarray, threshold: float, k: int = 3, seed: int = 42):
    """Même protocole que arbre/ : on remplace les k premières raisons de chaque alerte par la médiane du train et
    on compte les alertes qui repassent sous le seuil. Témoin : k features tirées au hasard."""
    x = np.asarray(x, dtype=np.float32)
    with torch.no_grad():
        before = scorer(torch.from_numpy(x)).numpy()
    alerted = before >= threshold
    xa = x[alerted]
    top = np.argsort(-contributions(scorer, xa), axis=1)[:, :k]
    rnd = np.random.default_rng(seed).random((len(xa), x.shape[1])).argsort(axis=1)[:, :k]

    def after(idx: np.ndarray) -> np.ndarray:
        xx = xa.copy()
        rows = np.arange(len(xx))[:, None]
        xx[rows, idx] = median[idx]
        with torch.no_grad():
            return scorer(torch.from_numpy(xx)).numpy()

    a_top, a_rnd = after(top), after(rnd)
    return {
        "k": k,
        "n_alerts": int(alerted.sum()),
        "share_below_threshold_after": float((a_top < threshold).mean()),
        "median_score_drop": float(np.median(before[alerted] - a_top)),
        "random_k_share_below_threshold": float((a_rnd < threshold).mean()),
        "random_k_median_score_drop": float(np.median(before[alerted] - a_rnd)),
    }
