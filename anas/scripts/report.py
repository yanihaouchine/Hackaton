"""Génère les rapports des alertes les plus prioritaires et mesure la latence d'un rapport.

Usage : python scripts/report.py --split test [--top 20] [--writer template|llm]
        python scripts/report.py --example   (rapport de l'exemple du contrat, sans données)
Sorties : <data>/reports/<alert_id>.md, <data>/reports/index.md, results/report_bench_<writer>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import polars as pl

from aml.data import STREAM_COLUMNS
from aml.dossier import CONTEXT_H, build_dossier
from aml.meta import max_rss_mb, run_meta
from aml.render import render_report
from aml.schema import AccountRef, Alert, AlertDossier, SubjectKyc, Transaction
from aml.typologies import TYPOLOGIES
from aml.writers import get_writer

EXAMPLE = ROOT / "contracts" / "examples" / "fanout_dossier.json"


def load_kyc(accounts_csv: Path, subjects: list[AccountRef]) -> dict[AccountRef, SubjectKyc]:
    """Les codes banque de accounts.csv n'ont pas de zéros en tête : on normalise avant la jointure."""
    acc = pl.scan_csv(accounts_csv, infer_schema=False).rename(
        {
            "Bank Name": "bank_name",
            "Bank ID": "bank_id",
            "Account Number": "account",
            "Entity ID": "entity_id",
            "Entity Name": "entity_name",
        }
    )
    wanted = {(str(int(s.bank)), s.account): s for s in subjects}
    found = acc.filter(pl.col("account").is_in([a for _, a in wanted])).collect()
    found = found.filter(
        pl.struct("bank_id", "account").map_elements(
            lambda r: (r["bank_id"], r["account"]) in wanted, return_dtype=pl.Boolean
        )
    )
    entities = found["entity_id"].unique().to_list()
    siblings = acc.filter(pl.col("entity_id").is_in(entities)).collect()
    out = {}
    for row in found.iter_rows(named=True):
        s = wanted[(row["bank_id"], row["account"])]
        others = siblings.filter(
            (pl.col("entity_id") == row["entity_id"]) & (pl.col("account") != s.account)
        )
        out[s] = SubjectKyc(
            bank_name=row["bank_name"],
            entity_id=row["entity_id"],
            entity_name=row["entity_name"],
            other_accounts=tuple(
                AccountRef(b.zfill(3), a)
                for b, a in others.select("bank_id", "account").head(10).iter_rows()
            ),
        )
    return out


def fetch_tx(tx_path: Path, alert: Alert) -> pl.DataFrame:
    """Transactions utiles au dossier, toutes antérieures ou égales au déclenchement."""
    s, t = alert.subject, alert.triggered_at
    lo = min(t - timedelta(hours=CONTEXT_H), t - timedelta(days=3))
    return (
        pl.scan_parquet(tx_path)
        .select(STREAM_COLUMNS)
        .filter(pl.col("ts").is_between(lo, t, closed="both"))
        .filter(
            pl.col("tx_id").is_in(list(alert.tx_ids))
            | ((pl.col("from_bank") == s.bank) & (pl.col("from_account") == s.account))
            | ((pl.col("to_bank") == s.bank) & (pl.col("to_account") == s.account))
        )
        .collect()
    )


def example_row(t: Transaction) -> dict[str, object]:
    d = t.to_dict()
    return {
        **{
            k: d[k]
            for k in (
                "tx_id",
                "amount_paid",
                "currency_paid",
                "amount_received",
                "currency_received",
                "payment_format",
            )
        },
        "ts": t.ts,
        "from_bank": t.src.bank,
        "from_account": t.src.account,
        "to_bank": t.dst.bank,
        "to_account": t.dst.account,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--data", type=Path, default=Path("../data/anas"))
    p.add_argument("--raw", type=Path, default=Path("../data/raw"))
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--writer", choices=["template", "llm"], default="template")
    p.add_argument("--example", action="store_true", help="rapport de l'exemple du contrat")
    args = p.parse_args()
    writer = get_writer(args.writer)
    out_dir = args.data / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    fx_path = args.data / "fx_usd.json"
    rates = (
        json.loads(fx_path.read_text(encoding="utf-8")) if fx_path.exists() else {"US Dollar": 1.0}
    )

    if args.example:  # faits recalculés par le code à partir des transactions de l'exemple
        given = AlertDossier.from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))
        tx = pl.DataFrame([example_row(t) for t in given.transactions])
        dossier = build_dossier(given.alert, tx, given.subject_kyc, rates)
        path = out_dir / "exemple_fanout.md"
        path.write_text(render_report(dossier, writer), encoding="utf-8")
        print(f"rapport -> {path}")
        return

    with open(args.data / f"alerts_{args.split}.jsonl", encoding="utf-8") as f:
        alerts = [Alert.from_dict(json.loads(line)) for line in f]
    alerts = sorted(alerts, key=lambda a: (a.priority_rank, -a.score, a.triggered_at))[: args.top]
    kyc = load_kyc(args.raw / "HI-Small_accounts.csv", [a.subject for a in alerts])

    fetch_ms, build_ms, render_ms, index = [], [], [], []
    for a in alerts:
        t0 = time.perf_counter()
        tx = fetch_tx(args.data / "tx.parquet", a)
        t1 = time.perf_counter()
        dossier = build_dossier(a, tx, kyc.get(a.subject), rates)
        t2 = time.perf_counter()
        md = render_report(dossier, writer)
        t3 = time.perf_counter()
        fetch_ms.append((t1 - t0) * 1e3)
        build_ms.append((t2 - t1) * 1e3)
        render_ms.append((t3 - t2) * 1e3)
        name = f"{a.alert_id}.md"
        (out_dir / name).write_text(md, encoding="utf-8")
        index.append(
            f"| {a.triggered_at:%d/%m %H:%M} | {a.priority_rank} | {a.score} | "
            f"{TYPOLOGIES[a.typology].name} | `{a.subject.key()}` | [{a.alert_id}]({name}) |"
        )
    (out_dir / "index.md").write_text(
        "# Alertes prioritaires\n\n| Date | Rang | Score | Typologie | Compte | Rapport |\n"
        "|---|---|---|---|---|---|\n" + "\n".join(index) + "\n",
        encoding="utf-8",
    )

    def pct(xs: list[float], q: int) -> float:
        return round(statistics.quantiles(xs, n=100)[q - 1], 2) if len(xs) >= 2 else round(xs[0], 2)

    bench = {
        "meta": run_meta(),
        "writer": writer.name,
        "n_reports": len(alerts),
        "dossier_plus_render_ms_p50": pct([b + r for b, r in zip(build_ms, render_ms)], 50),
        "dossier_plus_render_ms_p99": pct([b + r for b, r in zip(build_ms, render_ms)], 99),
        "fetch_ms_p50": pct(fetch_ms, 50),
        "fetch_ms_p99": pct(fetch_ms, 99),
        "max_rss_mb": round(max_rss_mb()),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"report_bench_{writer.name}.json").write_text(
        json.dumps(bench, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in bench.items() if k != "meta"}, indent=2))
    print(f"rapports -> {out_dir}/index.md")


if __name__ == "__main__":
    main()
