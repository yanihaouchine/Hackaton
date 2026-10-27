"""Dossier de faits d'un cas : opérations, comptes (KYC), preuves structurelles, raisons, typologie.

Les preuves sont reconstituées à partir des seules transactions antérieures ou égales à l'alerte (index ≤ i
dans le flux trié) : le dossier ne contient aucune information future ni aucune étiquette.
Tous les nombres, dates et identifiants du rapport proviennent de ce dossier (jamais du LLM).
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from aml.config import CURRENCIES, FORMATS, TYPOLOGY_FR

ACH = FORMATS.index("ACH")
ISO = {"US Dollar": "USD", "Euro": "EUR", "Yuan": "CNY", "Swiss Franc": "CHF", "UK Pound": "GBP", "Rupee": "INR",
       "Shekel": "ILS", "Yen": "JPY", "Ruble": "RUB", "Canadian Dollar": "CAD", "Bitcoin": "BTC",
       "Australian Dollar": "AUD", "Mexican Peso": "MXN", "Saudi Riyal": "SAR", "Brazil Real": "BRL"}  # fmt: skip


class History:
    """Index par compte (CSR) des transactions du flux, en ordre chronologique."""

    def __init__(self, stream, hi: int | None = None):
        self.s = stream
        hi = stream.n if hi is None else hi
        src = np.asarray(stream.src[:hi])
        dst = np.asarray(stream.dst[:hi])
        n_acc = len(stream.acc_etype)
        self.out_order = np.argsort(src, kind="stable").astype(np.int64)
        self.in_order = np.argsort(dst, kind="stable").astype(np.int64)
        self.out_ptr = np.r_[0, np.cumsum(np.bincount(src, minlength=n_acc))]
        self.in_ptr = np.r_[0, np.cumsum(np.bincount(dst, minlength=n_acc))]
        self.t = np.asarray(stream.t[:hi])
        self.fmt = np.asarray(stream.fmt[:hi])
        self.loop = src == dst  # auto-virements (réinvestissements) : hors graphe, comme dans le moteur

    def _range(self, order, ptr, acc, t_lo, i_max, ach_only):
        rows = order[ptr[acc] : ptr[acc + 1]]  # indices croissants = ordre chronologique
        rows = rows[(rows <= i_max) & (self.t[rows] >= t_lo) & ~self.loop[rows]]
        if ach_only:
            rows = rows[self.fmt[rows] == ACH]
        return rows

    def out_tx(self, acc: int, t_lo: int, i_max: int, ach_only: bool = True) -> np.ndarray:
        return self._range(self.out_order, self.out_ptr, acc, t_lo, i_max, ach_only)

    def in_tx(self, acc: int, t_lo: int, i_max: int, ach_only: bool = True) -> np.ndarray:
        return self._range(self.in_order, self.in_ptr, acc, t_lo, i_max, ach_only)


def find_cycle(h: History, i: int, window: int = 5760, max_len: int = 10, budget: int = 5000) -> list[int]:
    """Plus court circuit temporel dst → … → src refermé par la transaction i (montants à ±30 %)."""
    s = h.s
    a, d, t, x = int(s.src[i]), int(s.dst[i]), int(s.t[i]), float(s.usd_paid[i])
    lo, hi = x / 1.3, x * 1.3
    best: list[int] = []
    count = [budget]

    def dfs(node, t_prev, path_nodes, path_tx):
        nonlocal best
        if count[0] <= 0 or (best and len(path_tx) + 1 >= len(best)):
            return
        for j in h.out_tx(node, t_prev, i - 1):
            count[0] -= 1
            if count[0] <= 0:
                return
            u = float(s.usd_paid[j])
            if u < lo or u > hi or s.t[j] < t_prev:
                continue
            y = int(s.dst[j])
            if y == a:
                if not best or len(path_tx) + 2 < len(best):
                    best = [*path_tx, int(j), i]
                continue
            if len(path_tx) + 3 > max_len or y in path_nodes:
                continue
            dfs(y, int(s.t[j]), path_nodes | {y}, [*path_tx, int(j)])

    dfs(d, t - window, {d}, [])
    return best


def pass_through_trail(h: History, i: int, window: int = 5760, tol: float = 0.10, depth: int = 8) -> list[int]:
    """Remonte la chaîne de fonds de passage : entrée de montant quasi identique, puis sa propre entrée…"""
    s = h.s
    trail, cur = [], i
    for _ in range(depth):
        a, t, x = int(s.src[cur]), int(s.t[cur]), float(s.usd_paid[cur])
        ins = h.in_tx(a, t - window, cur - 1)
        if not len(ins):
            break
        err = np.abs(x / np.maximum(np.asarray(s.usd_recv[ins], dtype=np.float64), 1e-9) - 1.0)
        k = int(np.argmin(err[::-1]))  # à égalité, la plus récente
        j = int(ins[::-1][k])
        if err[::-1][k] > tol:
            break
        trail.append(j)
        cur = j
    return trail[::-1]


def tx_record(s, j: int, t0: datetime, names: dict[int, str]) -> dict:
    return {
        "index": int(j),
        "date": (t0 + timedelta(minutes=int(s.t[j]))).strftime("%d/%m/%Y %H:%M"),
        "src": names.get(int(s.src[j]), str(int(s.src[j]))),
        "dst": names.get(int(s.dst[j]), str(int(s.dst[j]))),
        "amount": float(s.amt_paid[j]),
        "currency": ISO[CURRENCIES[int(s.cur_paid[j])]],
        "amount_recv": float(s.amt_recv[j]),
        "currency_recv": ISO[CURRENCIES[int(s.cur_recv[j])]],
        "usd": float(s.usd_paid[j]),
        "format": FORMATS[int(s.fmt[j])],
    }


def evidence(h: History, i: int, window: int = 5760) -> dict:
    """Preuves structurelles d'une alerte (transactions d'index ≤ i uniquement)."""
    s = h.s
    a, d, t = int(s.src[i]), int(s.dst[i]), int(s.t[i])
    fan_out = h.out_tx(a, t - window, i)
    fan_in = h.in_tx(d, t - window, i)
    src_in = h.in_tx(a, t - window, i - 1)
    srcs = list(dict.fromkeys(int(x) for x in s.src[src_in]))
    targets_of_d = set(int(x) for x in s.src[fan_in])
    sg = []
    for s0 in srcs[:64]:
        for j in h.out_tx(s0, t - window, i):
            w = int(s.dst[j])
            if w not in (a, d) and w in targets_of_d:
                sg.append(int(j))
    return {
        "fan_out_tx": fan_out.tolist(),
        "fan_out_distinct": int(len(set(s.dst[fan_out].tolist()))),
        "fan_in_tx": fan_in.tolist(),
        "fan_in_distinct": int(len(set(s.src[fan_in].tolist()))),
        "pass_through_trail": pass_through_trail(h, i, window),
        "cycle_tx": find_cycle(h, i, window),
        "scatter_gather_tx": sg[:50],
    }


def account_facts(acc_table, acc_id: int, countries: list[str], stream) -> dict:
    row = acc_table[acc_id]
    return {
        "id": row["key"],
        "bank": row["bank_name"] or "inconnue",
        "country": countries[int(stream.acc_country[acc_id])],
        "entity": row["ent_name"] or "inconnue",
        "entity_accounts": int(stream.acc_nacc[acc_id]),
    }


def typology_label(code: str) -> str:
    return TYPOLOGY_FR.get(code, code)
