"""Rejoue le flux sur une période et écrit les alertes (format du contrat), avec les mesures de frugalité.

Usage : python scripts/detect.py --split test [--data ../data/anas]
Sorties : <data>/alerts_<split>.jsonl, results/detect_<split>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aml.config import SPLITS
from aml.engine import StreamDetector, Thresholds
from aml.meta import max_rss_mb, run_meta
from aml.schema import Alert
from aml.stream import replay


def rank_by_day(alerts: list[Alert]) -> list[Alert]:
    """Rang de priorité dans la journée : score décroissant (file de l'analyste en fin de journée)."""
    out, by_day = [], {}
    for a in alerts:
        by_day.setdefault(a.triggered_at.date(), []).append(a)
    for day in sorted(by_day):
        ranked = sorted(by_day[day], key=lambda a: (-a.score, a.triggered_at, a.alert_id))
        out.extend(replace(a, priority_rank=i) for i, a in enumerate(ranked, start=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=sorted(SPLITS), default="test")
    p.add_argument("--data", type=Path, default=Path("../data/anas"))
    p.add_argument("--thresholds", type=Path, default=ROOT / "config" / "thresholds.json")
    p.add_argument(
        "--warmup-h", type=float, default=72, help="historique rejoué avant la période, sans alerte"
    )
    p.add_argument(
        "--hours", type=float, default=None, help="limiter la période (échantillon rapide)"
    )
    args = p.parse_args()

    start, end = SPLITS[args.split]
    if args.hours:
        end = min(end, start + timedelta(hours=args.hours))
    th = Thresholds.load(args.thresholds)
    rates = json.loads((args.data / "fx_usd.json").read_text(encoding="utf-8"))
    det = StreamDetector(th, rates, emit_from=start)
    alerts, _, stats = replay(
        det, args.data / "tx.parquet", start - timedelta(hours=args.warmup_h), end
    )
    alerts = rank_by_day(alerts)

    path = args.data / f"alerts_{args.split}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(a.to_dict(), ensure_ascii=False) + "\n" for a in alerts)

    lat = sorted(stats.latencies_us)
    q = statistics.quantiles(lat, n=100) if len(lat) >= 100 else [float("nan")] * 99
    by_typ: dict[str, int] = {}
    for a in alerts:
        by_typ[a.typology] = by_typ.get(a.typology, 0) + 1
    out = {
        "meta": run_meta(),
        "split": args.split,
        "period": [start.isoformat(), end.isoformat()],
        "warmup_h": args.warmup_h,
        "n_tx_replayed": stats.n_tx,
        "seconds": round(stats.seconds, 2),
        "throughput_tx_s": round(stats.n_tx / stats.seconds),
        "latency_us_p50": round(q[49], 2),
        "latency_us_p99": round(q[98], 2),
        "max_rss_mb": round(max_rss_mb()),
        "state_at_end": det.state_size(),
        "n_alerts": len(alerts),
        "alerts_by_typology": by_typ,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"detect_{args.split}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=2))
    print(f"alertes -> {path}")


if __name__ == "__main__":
    main()
