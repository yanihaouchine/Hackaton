"""Le MLP et sa normalisation intégrée (exportée telle quelle en ONNX : l'inférence prend les features brutes)."""

from __future__ import annotations

import torch
from torch import nn

from nn.data import CLIP, Preprocessing

HIDDEN = (128, 64)
DROPOUT = 0.1


class Normalize(nn.Module):
    """Même calcul que `Preprocessing.transform`, en opérations torch exportables en ONNX."""

    def __init__(self, pre: Preprocessing):
        super().__init__()
        self.register_buffer("num_idx", torch.tensor(pre.num_idx, dtype=torch.long))
        self.register_buffer("mean", torch.tensor(pre.num_mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(pre.num_std, dtype=torch.float32))
        self.cats = [(j, k) for j, k in zip(pre.cat_idx, pre.cat_sizes, strict=True)]
        for j, k in self.cats:
            self.register_buffer(f"codes_{j}", torch.arange(k, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.index_select(1, self.num_idx)
        z = torch.clamp((torch.sign(v) * torch.log1p(torch.abs(v)) - self.mean) / self.std, -CLIP, CLIP)
        parts = [z] + [(x[:, j : j + 1] == getattr(self, f"codes_{j}")).float() for j, _ in self.cats]
        return torch.cat(parts, dim=1)


def mlp(n_inputs: int, hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT) -> nn.Sequential:
    """n_inputs → 128 → 64 → 1 (logit). Le dropout n'agit qu'à l'entraînement."""
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(n_inputs, h), nn.ReLU(), nn.Dropout(dropout)]
        n_inputs = h
    layers.append(nn.Linear(n_inputs, 1))
    return nn.Sequential(*layers)


class Scorer(nn.Module):
    """Modèle livré : features brutes (colonnes du modèle strict) → probabilité d'être illicite."""

    def __init__(self, pre: Preprocessing, net: nn.Sequential):
        super().__init__()
        self.norm = Normalize(pre)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(self.norm(x))).squeeze(1)


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
