"""Mise en forme française des nombres, montants, dates et durées (faite par le code, jamais par un LLM)."""

from __future__ import annotations

from datetime import datetime

from aml.config import CURRENCY_CODES


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def fmt_amount(x: float, currency: str) -> str:
    whole, dec = f"{x:,.2f}".split(".")
    return f"{whole.replace(',', ' ')},{dec} {CURRENCY_CODES.get(currency, currency)}"


def fmt_dt(dt: datetime) -> str:
    return f"{dt:%d/%m/%Y} à {dt:%H} h {dt:%M}"


def fmt_duration(minutes: int) -> str:
    days, rest = divmod(minutes, 24 * 60)
    hours, mins = divmod(rest, 60)
    hm = f"{hours} h {mins:02d}"
    return f"{days} j {hm}" if days else hm


def fmt_ratio(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")
