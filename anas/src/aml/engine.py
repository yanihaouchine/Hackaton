"""Détection en flux par règles explicables.

Chaque transaction est traitée dans l'ordre chronologique, avec seulement l'état construit
à partir des transactions antérieures (flux causal). L'état est borné : fenêtres glissantes
de `window_h` heures par compte, purgées régulièrement.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path

from aml.schema import AccountRef, Alert, Reason

EPOCH = datetime(1970, 1, 1)
MAX_TX_PER_ALERT = 50
MAX_CHAIN = 10


@dataclass(frozen=True)
class Thresholds:
    window_h: float = 24.0
    fan_out: int = 6
    fan_in: int = 6
    chain_depth: int = 4
    chain_min_ratio: float = 0.8
    structuring_limit_usd: float = 10_000.0
    structuring_band: float = 0.1
    structuring_count: int = 3
    dormant_gap_h: float = 48.0
    dormant_min_usd: float = 50_000.0
    cooldown_h: float = 24.0
    # Formats de paiement sur lesquels les règles se déclenchent (vide = tous).
    # Fixés sur l'entraînement : formats dont le taux d'illicites dépasse 3 fois la moyenne.
    risky_formats: tuple[str, ...] = ()

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["risky_formats"] = tuple(kwargs.get("risky_formats", ()))
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["risky_formats"] = list(self.risky_formats)
        return d


class _Window:
    """Transactions d'un compte dans la fenêtre glissante, d'un côté (entrant ou sortant)."""

    __slots__ = ("counts", "items", "n_struct")

    def __init__(self) -> None:
        self.items: deque[tuple[int, int, int, bool]] = (
            deque()
        )  # (minute, contrepartie, tx_id, fractionné)
        self.counts: dict[int, int] = {}
        self.n_struct = 0

    def add(self, t: int, cp: int, tx_id: int, struct: bool) -> None:
        self.items.append((t, cp, tx_id, struct))
        self.counts[cp] = self.counts.get(cp, 0) + 1
        self.n_struct += struct

    def expire(self, cutoff: int) -> None:
        items, counts = self.items, self.counts
        while items and items[0][0] < cutoff:
            _, cp, _, struct = items.popleft()
            n = counts[cp] - 1
            if n:
                counts[cp] = n
            else:
                del counts[cp]
            self.n_struct -= struct

    def tx_ids(self, struct_only: bool = False) -> list[int]:
        return [i for _, _, i, s in self.items if s or not struct_only][-MAX_TX_PER_ALERT:]


@dataclass(frozen=True)
class _Chain:
    """Dernier passage de fonds entrant sur un compte : profondeur de la chaîne qui l'a amené."""

    depth: int
    origin: int
    start: int
    t: int
    usd: float
    path: tuple[int, ...]


@dataclass
class Features:
    """Indicateurs causaux calculés pour une transaction (servent au réglage des seuils)."""

    tx_id: int
    fan_out: int
    fan_in: int
    src_fan_in: int
    chain_depth: int
    cycle: bool
    n_struct: int
    gap_h: float
    usd: float
    risky: bool


class StreamDetector:
    def __init__(
        self, th: Thresholds, usd_rates: dict[str, float], emit_from: datetime | None = None
    ):
        self.th = th
        self.rates = usd_rates
        self.window = int(th.window_h * 60)
        self.emit_from = _minutes(emit_from) if emit_from else None
        self.ids: dict[tuple[str, str], int] = {}
        self.refs: list[tuple[str, str]] = []
        self.out: dict[int, _Window] = {}
        self.inc: dict[int, _Window] = {}
        self.chain: dict[int, _Chain] = {}
        self.last_seen: dict[int, int] = {}
        self.cooldown: dict[tuple[int, str], int] = {}
        self.next_sweep = -1
        self.n_alerts = 0
        lo = th.structuring_limit_usd * (1 - th.structuring_band)
        self.struct_band = (lo, th.structuring_limit_usd)

    # -- état ---------------------------------------------------------------
    def _acc(self, bank: str, account: str) -> int:
        key = (bank, account)
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.refs)
            self.refs.append(key)
        return i

    def _win(self, table: dict[int, _Window], acc: int, cutoff: int) -> _Window:
        w = table.get(acc)
        if w is None:
            w = table[acc] = _Window()
        else:
            w.expire(cutoff)
        return w

    def _sweep(self, t: int) -> None:
        """Purge des états expirés : la mémoire reste proportionnelle à l'activité récente."""
        cutoff = t - self.window
        for table in (self.out, self.inc):
            for acc in [a for a, w in table.items() if (w.expire(cutoff), not w.items)[1]]:
                del table[acc]
        self.chain = {a: c for a, c in self.chain.items() if c.t >= cutoff}
        cd = int(self.th.cooldown_h * 60)
        self.cooldown = {k: v for k, v in self.cooldown.items() if v > t - cd}
        self.next_sweep = t + 6 * 60

    def state_size(self) -> dict[str, int]:
        return {
            "accounts_known": len(self.refs),
            "windows_out": len(self.out),
            "windows_in": len(self.inc),
            "window_items": sum(len(w.items) for w in self.out.values())
            + sum(len(w.items) for w in self.inc.values()),
            "chains": len(self.chain),
        }

    # -- traitement d'une transaction ----------------------------------------
    def process(
        self,
        tx_id: int,
        t: int,
        from_bank: str,
        from_account: str,
        to_bank: str,
        to_account: str,
        amount_paid: float,
        currency_paid: str,
        payment_format: str,
    ) -> tuple[Features | None, list[Alert]]:
        if t >= self.next_sweep:
            self._sweep(t)
        th = self.th
        src = self._acc(from_bank, from_account)
        dst = self._acc(to_bank, to_account)
        usd = amount_paid * self.rates.get(currency_paid, 0.0)
        prev_seen = self.last_seen.get(src)
        self.last_seen[src] = t
        self.last_seen[dst] = t
        if src == dst:  # auto-virement : pas d'arête dans le graphe
            return None, []

        cutoff = t - self.window
        lo, hi = self.struct_band
        struct = lo <= usd < hi
        w_out = self._win(self.out, src, cutoff)
        w_out.add(t, dst, tx_id, struct)
        w_in = self._win(self.inc, dst, cutoff)
        w_in.add(t, src, tx_id, False)
        src_in = self.inc.get(src)
        if src_in is not None:
            src_in.expire(cutoff)
        src_fan_in = len(src_in.counts) if src_in is not None else 0

        # Chaîne : les fonds reçus par src sont-ils réexpédiés (presque en totalité) ?
        prev = self.chain.get(src)
        if (
            prev is not None
            and t - prev.start <= self.window
            and th.chain_min_ratio * prev.usd <= usd <= prev.usd * 1.05
        ):
            link = _Chain(
                prev.depth + 1, prev.origin, prev.start, t, usd, (*prev.path, tx_id)[-MAX_CHAIN:]
            )
        else:
            link = _Chain(1, src, t, t, usd, (tx_id,))
        cur = self.chain.get(dst)
        if cur is None or cur.t < cutoff or link.depth >= cur.depth:
            self.chain[dst] = link
        cycle = link.depth >= 3 and dst == link.origin

        gap_h = (t - prev_seen) / 60 if prev_seen is not None else -1.0
        feats = Features(
            tx_id,
            len(w_out.counts),
            len(w_in.counts),
            src_fan_in,
            link.depth,
            cycle,
            w_out.n_struct,
            gap_h,
            usd,
            not th.risky_formats or payment_format in th.risky_formats,
        )
        if (self.emit_from is not None and t < self.emit_from) or not feats.risky:
            return feats, []

        # Règles -> raisons par compte concerné.
        hits: dict[int, list[tuple[str, Reason, list[int]]]] = {}
        wh = th.window_h
        if cycle:
            r = Reason("CYCLE_CLOSED", link.depth, 3, wh)
            hits.setdefault(dst, []).append(("cycle", r, list(link.path)))
        elif link.depth >= th.chain_depth:
            r = Reason("RAPID_CHAIN_DEPTH", link.depth, th.chain_depth, wh)
            hits.setdefault(dst, []).append(("rapid_chain", r, list(link.path)))
        if feats.fan_out >= th.fan_out:
            r = Reason("FANOUT_DISTINCT_DEST", feats.fan_out, th.fan_out, wh)
            typ = "gather_scatter" if src_fan_in >= th.fan_in else "fan_out"
            hits.setdefault(src, []).append((typ, r, w_out.tx_ids()))
        if feats.fan_in >= th.fan_in:
            r = Reason("FANIN_DISTINCT_SRC", feats.fan_in, th.fan_in, wh)
            hits.setdefault(dst, []).append(("fan_in", r, w_in.tx_ids()))
        if w_out.n_struct >= th.structuring_count:
            r = Reason("STRUCTURING_BELOW_LIMIT", w_out.n_struct, th.structuring_count, wh)
            hits.setdefault(src, []).append(("structuring", r, w_out.tx_ids(struct_only=True)))
        if gap_h >= th.dormant_gap_h and usd >= th.dormant_min_usd:
            r = Reason("DORMANT_REACTIVATED", round(gap_h, 1), th.dormant_gap_h, None)
            hits.setdefault(src, []).append(("dormant", r, [tx_id]))

        alerts = []
        cd = int(th.cooldown_h * 60)
        for acc, found in hits.items():
            fresh = [h for h in found if self.cooldown.get((acc, h[0]), -cd) <= t - cd]
            if not fresh:
                continue
            for typ, _, _ in fresh:
                self.cooldown[(acc, typ)] = t
            alerts.append(self._alert(acc, t, tx_id, fresh))
        return feats, alerts

    def _alert(
        self, acc: int, t: int, tx_id: int, found: list[tuple[str, Reason, list[int]]]
    ) -> Alert:
        typology = found[0][0]
        tx_ids = sorted({i for _, _, ids in found for i in ids} | {tx_id})
        score = sum(r.value / r.threshold for _, r, _ in found if r.threshold)
        bank, account = self.refs[acc]
        when = EPOCH + timedelta(minutes=t)
        self.n_alerts += 1
        return Alert(
            alert_id=f"A-{when:%Y%m%dT%H%M}-{bank}-{account}-{typology}",
            triggered_at=when,
            subject=AccountRef(bank, account),
            trigger_tx_id=tx_id,
            tx_ids=tuple(tx_ids),
            score=round(score, 3),
            score_kind="rule",
            priority_rank=1,  # fixé ensuite par jour, par score décroissant
            typology=typology,
            reasons=tuple(r for _, r, _ in found),
            detector_name="anas-rules",
            detector_version="0.1",
        )


def _minutes(dt: datetime) -> int:
    return int((dt - EPOCH).total_seconds() // 60)
