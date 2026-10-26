"""Choix des seuils des règles sur la VALIDATION uniquement (critère : F1 de chaque règle seule).

Usage : python scripts/tune.py [--data ../data/anas]
Sorties : config/thresholds.json (seuils figés), results/tune.json (détail de la grille).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl

from aml.config import SPLITS
from aml.engine import StreamDetector, Thresholds
from aml.meta import run_meta
from aml.metrics import prf
from aml.stream import replay

GRIDS = {
    "fan_out": ("fan_out", range(2, 16)),
    "fan_in": ("fan_in", range(2, 16)),
    "chain_depth": ("chain_depth", range(2, 9)),
    "structuring_count": ("n_struct", range(1, 9)),
}


def best(feats: pl.DataFrame, col: str, grid: range, n_pos: int) -> tuple[int, list[dict]]:
    rows = []
    for th in grid:
        hit = feats.filter(pl.col(col) >= th)
        m = prf(int(hit["is_laundering"].sum()), hit.height, n_pos)
        rows.append({"threshold": th, "n_flagged": hit.height, **m})
    return max(rows, key=lambda r: r["f1"])["threshold"], rows


def risky_formats(path: Path, factor: float = 3.0) -> tuple[str, ...]:
    """Formats dont le taux d'illicites sur l'ENTRAÎNEMENT dépasse `factor` fois la moyenne."""
    start, end = SPLITS["train"]
    train = (
        pl.scan_parquet(path)
        .filter(pl.col("ts").is_between(start, end, closed="left"))
        .select("payment_format", "is_laundering")
        .collect()
    )
    base = train["is_laundering"].mean()
    rates = train.group_by("payment_format").agg(pl.col("is_laundering").mean().alias("rate"))
    return tuple(sorted(rates.filter(pl.col("rate") > factor * base)["payment_format"].to_list()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("../data/anas"))
    p.add_argument("--warmup-h", type=float, default=72)
    args = p.parse_args()

    risky = risky_formats(args.data / "tx.parquet")
    start, end = SPLITS["val"]
    rates = json.loads((args.data / "fx_usd.json").read_text(encoding="utf-8"))
    det = StreamDetector(
        Thresholds(risky_formats=risky), rates, emit_from=end
    )  # pas d'alerte : on veut les indicateurs
    _, feats, stats = replay(
        det, args.data / "tx.parquet", start - timedelta(hours=args.warmup_h), end, True
    )
    df = pl.DataFrame([asdict(f) for f in feats])
    labels = (
        pl.scan_parquet(args.data / "tx.parquet")
        .filter(pl.col("ts").is_between(start, end, closed="left"))
        .select("tx_id", "is_laundering")
        .collect()
    )
    df = df.join(labels, on="tx_id", how="inner")  # garde la période de validation seulement
    n_pos = int(labels["is_laundering"].sum())
    df = df.filter(pl.col("risky"))  # les règles ne se déclenchent que sur ces formats

    chosen: dict[str, float] = {}
    detail: dict[str, list[dict]] = {}
    for name, (col, grid) in GRIDS.items():
        chosen[name], detail[name] = best(df, col, grid, n_pos)

    dormant = []
    for gap in (24, 48, 72):
        for amount in (1e4, 5e4, 1e5, 1e6):
            hit = df.filter((pl.col("gap_h") >= gap) & (pl.col("usd") >= amount))
            m = prf(int(hit["is_laundering"].sum()), hit.height, n_pos)
            dormant.append({"gap_h": gap, "min_usd": amount, "n_flagged": hit.height, **m})
    d_best = max(dormant, key=lambda r: r["f1"])
    detail["dormant"] = dormant

    th = replace(
        Thresholds(),
        fan_out=chosen["fan_out"],
        fan_in=chosen["fan_in"],
        chain_depth=chosen["chain_depth"],
        structuring_count=chosen["structuring_count"],
        dormant_gap_h=d_best["gap_h"],
        dormant_min_usd=d_best["min_usd"],
        risky_formats=risky,
    )
    (ROOT / "config").mkdir(exist_ok=True)
    (ROOT / "config" / "thresholds.json").write_text(
        json.dumps(th.to_dict(), indent=2), encoding="utf-8"
    )
    (ROOT / "results").mkdir(exist_ok=True)
    out = {
        "meta": run_meta(),
        "split": "val",
        "period": [start.isoformat(), end.isoformat()],
        "n_tx": labels.height,
        "n_positives": n_pos,
        "replay_seconds": stats.seconds,
        "thresholds": th.to_dict(),
        "grids": detail,
    }
    (ROOT / "results" / "tune.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(th.to_dict(), indent=2, ensure_ascii=False))
    for name, rows in detail.items():
        b = max(rows, key=lambda r: r["f1"])
        print(
            f"{name:18s} P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f} n={b['n_flagged']}"
        )


if __name__ == "__main__":
    main()
