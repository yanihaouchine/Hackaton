"""Prépare le flux trié, le rapprochement avec Patterns.txt et les taux de change internes.

Usage : python scripts/prepare_data.py [--raw ../data/raw] [--out ../data/anas]
Sorties (régénérables, hors git) : tx.parquet, attempts.parquet, fx_usd.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from aml.config import SPLITS
from aml.data import (
    estimate_usd_rates,
    match_attempts,
    read_patterns,
    read_raw_strings,
    to_sorted_stream,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("../data/raw"))
    p.add_argument("--out", type=Path, default=Path("../data/anas"))
    p.add_argument("--variant", default="HI-Small")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    raw = read_raw_strings(args.raw / f"{args.variant}_Trans.csv")
    stream = to_sorted_stream(raw)
    attempts = match_attempts(raw, stream, read_patterns(args.raw / f"{args.variant}_Patterns.txt"))
    del raw

    train_start, train_end = SPLITS["train"]
    train = stream.filter(pl.col("ts").is_between(train_start, train_end, closed="left"))
    rates = estimate_usd_rates(train)

    stream.drop("raw_row").write_parquet(args.out / "tx.parquet")
    attempts.write_parquet(args.out / "attempts.parquet")
    (args.out / "fx_usd.json").write_text(
        json.dumps(rates, indent=2, sort_keys=True), encoding="utf-8"
    )

    missing = sorted(set(stream["currency_paid"].unique()) - set(rates))
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"transactions : {stream.height}")
    print(f"illicites : {int(stream['is_laundering'].sum())}")
    print(f"rapprochées avec Patterns.txt : {attempts.height}")
    print(f"devises sans taux : {missing or 'aucune'}")
    print(f"durée : {time.perf_counter() - t0:.1f} s ; RSS max : {rss_mb:.0f} Mo")


if __name__ == "__main__":
    main()
