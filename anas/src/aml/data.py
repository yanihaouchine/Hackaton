"""Lecture des fichiers bruts IBM AML et préparation du flux trié."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from aml.config import PATTERN_TYPOLOGIES

# Le CSV a deux colonnes "Account" : on les renomme dès la lecture.
RAW_COLUMNS = [
    "ts",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "currency_received",
    "amount_paid",
    "currency_paid",
    "payment_format",
    "is_laundering",
]
# Clé de rapprochement entre Trans.csv et Patterns.txt (chaînes brutes).
MATCH_KEY = [
    "ts",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_paid",
    "currency_paid",
    "payment_format",
]
# Colonnes lues par le moteur de détection : jamais d'étiquette.
STREAM_COLUMNS = [
    "tx_id",
    "ts",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_paid",
    "currency_paid",
    "amount_received",
    "currency_received",
    "payment_format",
]
LABEL_COLUMNS = ["is_laundering"]


def read_raw_strings(path: Path) -> pl.DataFrame:
    """Tout en chaînes : les codes banque gardent leurs zéros en tête."""
    return pl.read_csv(path, has_header=True, new_columns=RAW_COLUMNS, infer_schema=False)


def to_sorted_stream(raw: pl.DataFrame) -> pl.DataFrame:
    """Typage, tri stable sur l'horodatage, puis tx_id = rang dans le flux."""
    return (
        raw.with_row_index("raw_row")
        .with_columns(
            pl.col("ts").str.strptime(pl.Datetime("us"), "%Y/%m/%d %H:%M"),
            pl.col("amount_paid", "amount_received").cast(pl.Float64),
            pl.col("is_laundering").cast(pl.Int8),
        )
        .sort("ts", maintain_order=True)
        .with_row_index("tx_id")
    )


def read_patterns(path: Path) -> pl.DataFrame:
    """Une ligne par transaction de Patterns.txt, avec son numéro de tentative et sa typologie."""
    rows: list[list[str]] = []
    attempt, typology = -1, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                name = line.split(" - ", 1)[1].split(":")[0].strip()
                typology = PATTERN_TYPOLOGIES[name]
            elif line.startswith("END LAUNDERING ATTEMPT"):
                typology = None
            elif line and typology is not None:
                rows.append([str(attempt), typology, *line.split(",")])
    return pl.DataFrame(
        rows, schema=["attempt_id", "typology", *RAW_COLUMNS], orient="row"
    ).with_columns(pl.col("attempt_id").cast(pl.Int32))


def match_attempts(raw: pl.DataFrame, stream: pl.DataFrame, patterns: pl.DataFrame) -> pl.DataFrame:
    """(tx_id, attempt_id, typology) pour les transactions décrites dans Patterns.txt."""
    keys = raw.with_row_index("raw_row").select("raw_row", *MATCH_KEY)
    ids = stream.select("raw_row", "tx_id")
    return (
        patterns.select("attempt_id", "typology", *MATCH_KEY)
        .join(keys, on=MATCH_KEY, how="inner")
        .join(ids, on="raw_row")
        .select("tx_id", "attempt_id", "typology")
        .unique("tx_id", keep="first", maintain_order=True)
        .sort("tx_id")
    )


def estimate_usd_rates(stream: pl.DataFrame) -> dict[str, float]:
    """Taux vers l'USD estimés sur les opérations avec change du jeu lui-même (pas de source externe).

    À appeler sur la période d'entraînement uniquement.
    """
    fx = stream.filter(pl.col("currency_paid") != pl.col("currency_received")).with_columns(
        (pl.col("amount_received") / pl.col("amount_paid")).alias("ratio")
    )
    to_usd = (
        fx.filter(pl.col("currency_received") == "US Dollar")
        .group_by("currency_paid")
        .agg(pl.col("ratio").median())
    )
    from_usd = (
        fx.filter(pl.col("currency_paid") == "US Dollar")
        .group_by("currency_received")
        .agg((1 / pl.col("ratio")).median())
    )
    rates = {"US Dollar": 1.0}
    for cur, r in from_usd.iter_rows():
        rates[cur] = r
    for cur, r in to_usd.iter_rows():  # sens direct prioritaire
        rates[cur] = r
    return rates
