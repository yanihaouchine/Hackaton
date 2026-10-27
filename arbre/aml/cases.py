"""Alertes → cas : deux alertes partageant un compte à moins de 96 h appartiennent au même cas (union-find).

Un cas est l'unité de travail de l'analyste et de la déclaration ; sa priorité combine les probabilités
calibrées de ses alertes (1 − Π(1 − pᵢ)), départagées par le montant total.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


@dataclass
class Case:
    case_id: str
    alerts: list[int]  # indices de transactions (flux trié)
    accounts: list[int]
    t_first: int
    t_last: int
    total_usd: float
    priority: float
    max_score: float
    typology: str = "OTHER"
    typology_proba: float = 0.0
    extra: dict = field(default_factory=dict)


def build_cases(idx: np.ndarray, src: np.ndarray, dst: np.ndarray, t: np.ndarray, usd: np.ndarray,
                p_cal: np.ndarray, window: int = 5760, prefix: str = "C") -> list[Case]:  # fmt: skip
    """idx : alertes triées par temps ; src/dst/t/usd/p_cal : valeurs alignées sur idx."""
    n = len(idx)
    uf = UnionFind(n)
    last: dict[int, tuple[int, int]] = {}  # compte → (alerte, temps) la plus récente
    for k in range(n):
        for acc in (int(src[k]), int(dst[k])):
            prev = last.get(acc)
            if prev is not None and t[k] - prev[1] <= window:
                uf.union(k, prev[0])
            last[acc] = (k, int(t[k]))
    groups: dict[int, list[int]] = {}
    for k in range(n):
        groups.setdefault(uf.find(k), []).append(k)
    cases = []
    for members in groups.values():
        m = np.array(members)
        accs = sorted(set(src[m].tolist()) | set(dst[m].tolist()))
        pr = float(1.0 - np.prod(1.0 - np.clip(p_cal[m], 0.0, 1.0)))
        cases.append(Case(case_id="", alerts=idx[m].tolist(), accounts=accs, t_first=int(t[m].min()),
                          t_last=int(t[m].max()), total_usd=float(usd[m].sum()), priority=pr,
                          max_score=float(p_cal[m].max())))  # fmt: skip
    cases.sort(key=lambda c: (-c.priority, -c.total_usd, c.t_first))
    for j, c in enumerate(cases):
        c.case_id = f"{prefix}{c.t_first // 1440:02d}-{j + 1:04d}"
    return cases
