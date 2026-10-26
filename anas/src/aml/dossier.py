"""Construction du dossier d'une alerte : tous les faits chiffrés sont calculés ici, une seule fois."""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

import polars as pl

from aml.config import CURRENCY_CODES, PAYMENT_FORMATS_FR
from aml.fmt import fmt_amount, fmt_dt, fmt_duration, fmt_int
from aml.schema import AccountRef, Alert, AlertDossier, Fact, SubjectKyc, Transaction

CONTEXT_H = 24
MAX_CONTEXT_TX = 60


def _num(x: float) -> str:
    return fmt_int(int(x)) if float(x).is_integer() else f"{x:.1f}".replace(".", ",")


def _transactions(df: pl.DataFrame) -> list[Transaction]:
    return [
        Transaction(
            tx_id=r["tx_id"],
            ts=r["ts"],
            src=AccountRef(r["from_bank"], r["from_account"]),
            dst=AccountRef(r["to_bank"], r["to_account"]),
            amount_paid=r["amount_paid"],
            currency_paid=r["currency_paid"],
            amount_received=r["amount_received"],
            currency_received=r["currency_received"],
            payment_format=r["payment_format"],
        )
        for r in df.iter_rows(named=True)
    ]


def build_dossier(
    alert: Alert, tx: pl.DataFrame, kyc: SubjectKyc | None, usd_rates: dict[str, float]
) -> AlertDossier:
    """`tx` : transactions candidates (colonnes du flux, sans étiquette).

    On ne garde que celles de l'alerte et l'activité du compte dans les 24 h précédentes,
    toutes antérieures ou égales au déclenchement (flux causal).
    """
    s = alert.subject
    t = alert.triggered_at
    past = tx.filter(pl.col("ts") <= t)
    involves = ((pl.col("from_bank") == s.bank) & (pl.col("from_account") == s.account)) | (
        (pl.col("to_bank") == s.bank) & (pl.col("to_account") == s.account)
    )
    context = past.filter(involves & (pl.col("ts") >= t - timedelta(hours=CONTEXT_H)))
    chosen = (
        pl.concat(
            [past.filter(pl.col("tx_id").is_in(list(alert.tx_ids))), context.tail(MAX_CONTEXT_TX)]
        )
        .unique("tx_id")
        .sort("ts", "tx_id")
    )
    transactions = _transactions(chosen)
    core = [x for x in transactions if x.tx_id in set(alert.tx_ids)]

    facts: list[Fact] = []

    def add(key: str, value: object, text: str) -> None:
        facts.append(Fact(f"F{len(facts) + 1}", key, value, text))  # type: ignore[arg-type]

    add("subject", s.key(), s.key())
    add("triggered_at", t.isoformat(), fmt_dt(t))
    add("priority_rank", alert.priority_rank, fmt_int(alert.priority_rank))
    add("score", alert.score, _num(alert.score))
    add("n_tx", len(core), fmt_int(len(core)))
    cps = {x.dst if x.src == s else x.src for x in core} - {s}
    add("n_counterparties", len(cps), fmt_int(len(cps)))
    if core:
        first, last = core[0].ts, core[-1].ts
        span = int((last - first).total_seconds() // 60)
        add("first_ts", first.isoformat(), fmt_dt(first))
        add("last_ts", last.isoformat(), fmt_dt(last))
        add("span_minutes", span, fmt_duration(span))
        totals: Counter[str] = Counter()
        for x in core:
            totals[x.currency_paid] += x.amount_paid
        for cur, amount in sorted(totals.items(), key=lambda kv: -kv[1] * usd_rates.get(kv[0], 0)):
            add(f"total_{CURRENCY_CODES.get(cur, cur)}", round(amount, 2), fmt_amount(amount, cur))
        usd = sum(a * usd_rates.get(c, 0.0) for c, a in totals.items())
        add("total_usd_estimate", round(usd, 2), fmt_amount(usd, "US Dollar"))
        biggest = max(core, key=lambda x: x.amount_paid * usd_rates.get(x.currency_paid, 0.0))
        add("max_tx", biggest.amount_paid, fmt_amount(biggest.amount_paid, biggest.currency_paid))
        formats = Counter(x.payment_format for x in core)
        add(
            "formats",
            ";".join(f"{f}:{n}" for f, n in formats.most_common()),
            ", ".join(f"{PAYMENT_FORMATS_FR.get(f, f)} ({n})" for f, n in formats.most_common()),
        )
    for i, r in enumerate(alert.reasons, start=1):
        add(f"reason{i}_value", r.value, _num(r.value))
        if r.threshold is not None:
            add(f"reason{i}_threshold", r.threshold, _num(r.threshold))
        if r.window_h is not None:
            add(f"reason{i}_window", r.window_h, f"{_num(r.window_h)} h")
    self_fx = [
        x
        for x in transactions
        if x.src == s and x.dst == s and x.currency_paid != x.currency_received
    ]
    add("n_self_fx", len(self_fx), fmt_int(len(self_fx)))
    if kyc is not None:
        add("n_other_accounts", len(kyc.other_accounts), fmt_int(len(kyc.other_accounts)))

    return AlertDossier(
        alert=alert,
        subject_kyc=kyc,
        transactions=tuple(transactions),
        facts=tuple(facts),
        history_until=t,
    )
