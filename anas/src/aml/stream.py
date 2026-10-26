"""Rejeu du flux : lecture par lots du parquet trié, sans jamais tout charger en mémoire."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from aml.data import STREAM_COLUMNS
from aml.engine import EPOCH, Features, StreamDetector
from aml.schema import Alert

LATENCY_SAMPLE_EVERY = 50


def iter_batches(
    path: Path, start: datetime, end: datetime, batch_rows: int = 100_000
) -> Iterator[pl.DataFrame]:
    """Lots de transactions de [start, end), dans l'ordre du flux. Les étiquettes ne sont pas lues."""
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch_rows, columns=STREAM_COLUMNS):
        df = pl.from_arrow(rb)
        assert isinstance(df, pl.DataFrame)
        if df.height == 0 or df["ts"][-1] < start:
            continue
        if df["ts"][0] >= end:
            break
        df = df.filter(pl.col("ts").is_between(start, end, closed="left"))
        yield df.with_columns(((pl.col("ts") - pl.lit(EPOCH)).dt.total_minutes()).alias("t"))


@dataclass
class RunStats:
    n_tx: int = 0
    seconds: float = 0.0
    latencies_us: list[float] = field(default_factory=list)


def replay(
    detector: StreamDetector,
    path: Path,
    start: datetime,
    end: datetime,
    collect_features: bool = False,
) -> tuple[list[Alert], list[Features], RunStats]:
    alerts: list[Alert] = []
    feats: list[Features] = []
    stats = RunStats()
    clock = time.perf_counter
    t0 = clock()
    for df in iter_batches(path, start, end):
        cols = [
            df[c].to_list()
            for c in (
                "tx_id",
                "t",
                "from_bank",
                "from_account",
                "to_bank",
                "to_account",
                "amount_paid",
                "currency_paid",
                "payment_format",
            )
        ]
        for i, row in enumerate(zip(*cols)):
            if i % LATENCY_SAMPLE_EVERY == 0:
                s = clock()
                f, a = detector.process(*row)
                stats.latencies_us.append((clock() - s) * 1e6)
            else:
                f, a = detector.process(*row)
            if a:
                alerts.extend(a)
            if collect_features and f is not None:
                feats.append(f)
        stats.n_tx += df.height
    stats.seconds = clock() - t0
    return alerts, feats, stats
