"""Produit de vraies alertes et leurs dossiers en rejouant une journée en flux avec la détection de arbre/.

Réutilise en lecture seule le moteur C, la forêt « strict » avec cascade, le regroupement en cas et le
dossier de faits de arbre/ (sa démo écrirait dans arbre/, ce script écrit dans data/neurones/alertes/).
Aucune étiquette n'est lue. Remplacer la détection (MLP, combinaison) ne change pas le format de sortie.
Usage (depuis neurones/) : ../.venv/bin/python -m rapport.make_alerts [--date 2022-09-09] [--cases 30]
Sorties : data/neurones/alertes/<date>/<cas>.json et index.json
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from datetime import datetime, timedelta

import numpy as np

from rapport import ALERTS_DIR


def main() -> None:
    from aml.cases import build_cases
    from aml.config import minutes_since
    from aml.data import AccountNames
    from aml.dossier import History
    from aml.report import Reporter
    from aml.stream import Monitor
    from aml.typology import TypologyModel

    a = argparse.ArgumentParser()
    a.add_argument("--date", default="2022-09-09", help="journée rejouée (période de test, déjà évaluée)")
    a.add_argument("--cases", type=int, default=30, help="nombre de dossiers écrits (cas les plus prioritaires)")
    args = a.parse_args()

    t_start = time.perf_counter()
    mon = Monitor("HI-Small", "strict", cascade=True)
    s = mon.s
    t0 = datetime.fromisoformat(s.meta["t0"])
    end = (datetime.fromisoformat(args.date) + timedelta(days=1)).date().isoformat()
    lo = int(np.searchsorted(s.t, minutes_since(t0, args.date)))
    hi = int(np.searchsorted(s.t, minutes_since(t0, end)))
    warm_s = mon.warm(lo, use_snapshot=False)  # historique antérieur, sans alerte ni écriture d'instantané

    alerts, scores, feats, engine_s = [], [], [], 0.0
    for ch in mon.replay(hi, chunk=8192):  # flux causal : chaque transaction est scorée à son arrivée
        engine_s += ch.seconds
        if len(ch.alerts):
            alerts.extend(ch.alerts.tolist())
            scores.extend(ch.scores.tolist())
            feats.append(ch.feats)
    idx, p = np.array(alerts, dtype=np.int64), np.array(scores)
    x = np.concatenate(feats) if feats else np.zeros((0, mon.eng.n_features), np.float32)
    cases = build_cases(idx, np.asarray(s.src[idx]), np.asarray(s.dst[idx]), np.asarray(s.t[idx]),
                        np.asarray(s.usd_paid[idx], dtype=np.float64), mon.b.calibrate(p))  # fmt: skip

    rep = Reporter(s, mon.b, History(s, hi), AccountNames("HI-Small"), typology=TypologyModel())
    pos = {int(i): k for k, i in enumerate(idx)}
    out = ALERTS_DIR / args.date
    out.mkdir(parents=True, exist_ok=True)
    index, facts_ms = [], []
    for rank, c in enumerate(cases[: args.cases], start=1):
        rows = [pos[i] for i in c.alerts]
        t = time.perf_counter()
        f = rep.facts(c, x[rows], p[rows], rank=rank)
        facts_ms.append((time.perf_counter() - t) * 1e3)
        f["detector"] = {"solution": "arbre/", "model": mon.b.name, "cascade": mon.cascade}
        (out / f"{c.case_id}.json").write_text(json.dumps(f, ensure_ascii=False, indent=1, default=str))
        index.append({"case_id": c.case_id, "rank": rank, "priority": c.priority, "priority_pct": f["priority_pct"],
                      "typology": f["typology"]["label"], "n_alerts": len(c.alerts),
                      "usd_fmt": f["totals"]["usd_fmt"], "main_account": f["accounts"][0]["id"],
                      "period_start": f["period"]["start"]})  # fmt: skip
    summary = {
        "date": args.date,
        "n_tx": hi - lo,
        "n_alerts": len(alerts),
        "n_cases": len(cases),
        "n_dossiers": len(index),
        "warm_seconds": round(warm_s, 1),
        "engine_seconds": round(engine_s, 2),
        "engine_tx_per_s": round((hi - lo) / max(engine_s, 1e-9)),
        "facts_ms_p50": round(float(np.median(facts_ms)), 1) if facts_ms else None,
        "total_seconds": round(time.perf_counter() - t_start, 1),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "cases": index,
    }
    (out / "index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2, ensure_ascii=False))
    print(f"dossiers -> {out}")


if __name__ == "__main__":
    main()
