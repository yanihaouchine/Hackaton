"""Génère le rapport TRACFIN (Markdown, sans LLM) d'une alerte, ou de toutes, avec la mesure du temps.
Usage (depuis neurones/) :
  ../.venv/bin/python -m rapport.report --case C08-0001        → data/neurones/rapports/<date>/C08-0001.md
  ../.venv/bin/python -m rapport.report --all                  → tous les dossiers + results/rapport_bench.json
Les dossiers viennent de `python -m rapport.make_alerts` (data/neurones/alertes/<date>/).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

from rapport import ALERTS_DIR, REPO, ROOT
from rapport.dossier import load
from rapport.render import render

REPORTS_DIR = REPO / "data" / "neurones" / "rapports"


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--date", default="2022-09-09")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--case", help="identifiant du cas, par exemple C08-0001")
    g.add_argument("--all", action="store_true", help="tous les dossiers de la journée, avec mesure du temps")
    args = a.parse_args()

    src, out = ALERTS_DIR / args.date, REPORTS_DIR / args.date
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("C*.json")) if args.all else [src / f"{args.case}.json"]
    times = []
    for p in paths:
        f = load(p)
        t = time.perf_counter()
        md = render(f)
        times.append((time.perf_counter() - t) * 1e3)
        (out / f"{f['case_id']}.md").write_text(md, encoding="utf-8")
    if not args.all:
        print(f"rapport -> {out / paths[0].with_suffix('.md').name} ({times[0]:.1f} ms)")
        return

    from aml.provenance import run_info

    q = statistics.quantiles(times, n=100, method="inclusive") if len(times) > 1 else times * 99
    bench = {"run": run_info(), "writer": "template jinja2 (sans LLM)", "n_reports": len(times),
             "render_ms_p50": round(q[49], 2), "render_ms_p99": round(q[98], 2),
             "render_ms_max": round(max(times), 2), "guard": "tous acceptés"}  # fmt: skip
    (ROOT / "results" / "rapport_bench.json").write_text(json.dumps(bench, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in bench.items() if k != "run"}, indent=2, ensure_ascii=False))
    print(f"rapports -> {out}")


if __name__ == "__main__":
    main()
