"""Évalue les alertes d'une période contre les étiquettes (et compare à des baselines).

Usage : python scripts/evaluate.py --split test [--k 50]
Sortie : results/eval_<split>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl

from aml.config import SPLITS
from aml.meta import run_meta
from aml.metrics import baseline_ach, evaluate
from aml.schema import Alert


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=sorted(SPLITS), default="test")
    p.add_argument("--data", type=Path, default=Path("../data/anas"))
    p.add_argument("--k", type=int, default=50, help="alertes traitées par jour par un analyste")
    args = p.parse_args()

    with open(args.data / f"alerts_{args.split}.jsonl", encoding="utf-8") as f:
        alerts = [Alert.from_dict(json.loads(line)) for line in f]
    # Même période que la détection (elle a pu être raccourcie avec --hours).
    run = json.loads((ROOT / "results" / f"detect_{args.split}.json").read_text(encoding="utf-8"))
    start, end = (datetime.fromisoformat(x) for x in run["period"])
    period = (
        pl.scan_parquet(args.data / "tx.parquet")
        .filter(pl.col("ts").is_between(start, end, closed="left"))
        .select("tx_id", "ts", "is_laundering", "payment_format")
        .collect()
    )
    attempts = pl.read_parquet(args.data / "attempts.parquet")

    res = evaluate(alerts, period, attempts, args.k)
    by_rule = {}
    for typ in sorted({a.typology for a in alerts}):
        sub = [a for a in alerts if a.typology == typ]
        m = evaluate(sub, period, attempts, args.k)["transactions"]
        by_rule[typ] = {k: m[k] for k in ("precision", "recall", "f1")}
    out = {
        "meta": run_meta(),
        "split": args.split,
        "period": [start.isoformat(), end.isoformat()],
        **res,
        "baselines": {
            "random_precision": res["transactions"]["prevalence"],
            "ach_only": baseline_ach(period),
            "each_typology_alone": by_rule,
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"eval_{args.split}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    t = res["transactions"]
    print(
        f"période {args.split} : {res['n_tx']} tx, {t['n_positives']} illicites, {res['n_alerts']} alertes"
    )
    print(
        f"transactions : P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} PR-AUC={t['pr_auc']:.3f}"
    )
    print(
        f"alertes justes : {res['alert_precision']:.3f} ; précision@{args.k}/jour : "
        f"{res['precision_at_k']['mean_over_days']:.3f}"
    )
    print(f"rappel hors motifs : {res['recall_outside_patterns']:.3f}")
    print("tentatives détectées par typologie :")
    for typ, d in res["attempts_detected_by_typology"].items():
        print(f"  {typ:15s} {d['detected']}/{d['attempts']} ({d['rate']:.0%})")
    b = out["baselines"]["ach_only"]
    print(f"baseline « tout ACH » : P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}")


if __name__ == "__main__":
    main()
