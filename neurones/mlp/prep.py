"""Prétraitement propre au MLP, ajusté sur le train uniquement puis figé dans le graphe du modèle.

Entrée : les features brutes du moteur (colonnes retenues, dans l'ordre du moteur).
Sortie : one-hot des catégories, heure en sinus/cosinus, indicateurs « absent » (-1) et « aucun appariement » (9),
log1p des compteurs et délais, standardisation et écrêtage à ±CLIP.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
from torch import nn

CLIP = 5.0
CATEGORICAL = ("tx_fmt", "tx_cur", "kyc_s_etype", "kyc_r_etype", "kyc_s_country", "kyc_r_country")
HOUR = "tx_hour"
# -1 = valeur absente (compte ou relation jamais vus, aucune chaîne).
MISSING = re.compile(r"(_age|_dt_out|_dt_in|_ach_out|_ach_in)$|^(p_dt|p_rev_dt|f_best_lag|f_chain_age)$")
# 9 = plafond « aucun appariement trouvé ».
NO_MATCH = ("f_best_err", "f_r_ret_err", "m_cyc_drift")
# Compteurs, délais et degrés à queue lourde.
LOG = re.compile(r"_n_|_ewc_|_dist_|_nacc$|_age$|_dt_|^p_|^g_|^f_best_lag$|^f_chain_age$|^m_(sg|bip|cyc4_n)$|_n_self$")


class Prep(nn.Module):
    """Transformations figées ; aucun paramètre appris par descente de gradient."""

    def __init__(self, names: list[str], cat_sizes: dict[str, int], mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.names = list(names)
        self.cat_sizes = dict(cat_sizes)
        idx = {n: i for i, n in enumerate(names)}
        self.cat = [(idx[n], cat_sizes[n]) for n in CATEGORICAL]
        self.hour = idx[HOUR]
        skip = set(CATEGORICAL) | {HOUR}
        num = [n for n in names if n not in skip]
        self.num_names = num
        self.register_buffer("num_idx", torch.tensor([idx[n] for n in num]))
        self.register_buffer("miss_mask", torch.tensor([bool(MISSING.search(n)) for n in num]))
        self.register_buffer("log_mask", torch.tensor([bool(LOG.search(n)) for n in num]))
        self.register_buffer("miss_idx", torch.tensor([i for i, n in enumerate(num) if MISSING.search(n)]))
        self.register_buffer("nomatch_idx", torch.tensor([num.index(n) for n in NO_MATCH]))
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def dense(self, x: torch.Tensor) -> torch.Tensor:
        """Partie numérique avant standardisation (sert aussi à calculer mean et std)."""
        v = x[:, self.num_idx]
        missing = (v == -1.0) & self.miss_mask
        v = torch.where(missing, torch.zeros_like(v), v)
        v = torch.where(self.log_mask, torch.log1p(torch.clamp(v, min=0.0)), v)
        miss_ind = missing[:, self.miss_idx].to(v.dtype)
        nomatch_ind = (x[:, self.num_idx][:, self.nomatch_idx] >= 9.0).to(v.dtype)
        return torch.cat([v, miss_ind, nomatch_ind], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = torch.clamp((self.dense(x) - self.mean) / self.std, -CLIP, CLIP)
        parts = [d]
        for i, size in self.cat:
            code = x[:, i : i + 1].round()
            parts.append((code == torch.arange(size, dtype=x.dtype, device=x.device)).to(x.dtype))
        angle = x[:, self.hour : self.hour + 1] * (2.0 * math.pi / 24.0)
        parts += [torch.sin(angle), torch.cos(angle)]
        return torch.cat(parts, dim=1)

    @property
    def out_dim(self) -> int:
        return int(self.mean.numel()) + sum(s for _, s in self.cat) + 2


def fit_prep(x_train: np.ndarray, names: list[str], batch: int = 262144) -> Prep:
    """Vocabulaires, moyennes et écarts-types calculés sur les lignes de train fournies."""
    idx = {n: i for i, n in enumerate(names)}
    cat_sizes = {n: int(np.max(x_train[:, idx[n]])) + 1 for n in CATEGORICAL}
    probe = Prep(names, cat_sizes, np.zeros(1), np.ones(1))
    s1 = s2 = None
    n = 0
    with torch.no_grad():
        for lo in range(0, len(x_train), batch):
            d = probe.dense(torch.from_numpy(np.ascontiguousarray(x_train[lo : lo + batch]))).double()
            s1 = d.sum(0) if s1 is None else s1 + d.sum(0)
            s2 = (d * d).sum(0) if s2 is None else s2 + (d * d).sum(0)
            n += len(d)
    mean = s1 / n
    std = torch.sqrt(torch.clamp(s2 / n - mean * mean, min=0.0))
    std = torch.where(std < 1e-6, torch.ones_like(std), std)  # colonne constante sur le train
    return Prep(names, cat_sizes, mean.numpy(), std.numpy())
