"""Génère le rapport TRACFIN (Markdown) d'une alerte, ou de toutes, avec la mesure du temps.
Usage (depuis neurones/) :
  ../.venv/bin/python -m rapport.report --case C08-0001        → data/neurones/rapports/<date>/C08-0001.md
  ../.venv/bin/python -m rapport.report --all                  → tous les dossiers + results/rapport_bench.json
  ../.venv/bin/python -m rapport.report --all --llm [--llm-model gemma4:e2b] [--llm-url http://localhost:11434]
                                         → résumé reformulé par le LLM local + results/rapport_bench_llm_<machine>.json
Les dossiers viennent de `python -m rapport.make_alerts` (data/neurones/alertes/<date>/).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

from rapport import ALERTS_DIR, REPO, ROOT
from rapport.dossier import load
from rapport.render import render

REPORTS_DIR = REPO / "data" / "neurones" / "rapports"


def pct(xs: list[float]) -> tuple[float, float]:
    q = statistics.quantiles(xs, n=100, method="inclusive") if len(xs) > 1 else xs * 99
    return round(q[49], 2), round(q[98], 2)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--date", default="2022-09-09")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--case", help="identifiant du cas, par exemple C08-0001")
    g.add_argument("--all", action="store_true", help="tous les dossiers de la journée, avec mesure du temps")
    a.add_argument("--llm", action="store_true", help="résumé reformulé par le LLM local, sous garde")
    a.add_argument("--llm-url", default=None)
    a.add_argument("--llm-model", default=None)
    args = a.parse_args()

    writer = None
    if args.llm:
        from rapport.llm import LlmUnavailable, OpenAICompatible

        try:
            writer = OpenAICompatible(args.llm_url, args.llm_model)
        except LlmUnavailable as e:
            sys.exit(f"LLM indisponible : {e}. Sur une machine de l'école : `ollama serve` puis relancer.")
        print(f"LLM : {writer.model} ({writer.url})")

    src, out = ALERTS_DIR / args.date, REPORTS_DIR / args.date
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("C*.json")) if args.all else [src / f"{args.case}.json"]
    times, accepted, reasons = [], 0, []
    for p in paths:
        f = load(p)
        t = time.perf_counter()
        md = render(f, writer)
        times.append((time.perf_counter() - t) * 1e3)
        if writer is not None:
            ok = "résumé reformulé par un LLM" in md
            accepted += ok
            if not ok:
                reasons.append(md.split("LLM non retenu : ", 1)[-1].split(")", 1)[0])
        suffix = "_llm" if writer is not None else ""
        (out / f"{f['case_id']}{suffix}.md").write_text(md, encoding="utf-8")
    if not args.all:
        print(f"rapport -> {out / (paths[0].stem + ('_llm' if writer else '') + '.md')} ({times[0]:.1f} ms)")
        return

    from aml.provenance import run_info

    p50, p99 = pct(times)
    bench = {"run": run_info(), "n_reports": len(times), "render_ms_p50": p50, "render_ms_p99": p99,
             "render_ms_max": round(max(times), 2)}  # fmt: skip
    if writer is None:
        bench.update(writer="template jinja2 (sans LLM)", guard="tous acceptés")
        name = "rapport_bench.json"
    else:
        bench.update(writer=f"LLM {writer.model} via {writer.url} (résumé seulement) + template",
                     llm_accepted=accepted, llm_acceptance_rate=accepted / len(times),
                     rejection_reasons=dict(sorted({r: reasons.count(r) for r in reasons}.items())))  # fmt: skip
        name = f"rapport_bench_llm_{platform.node()}.json"
    (ROOT / "results" / name).write_text(json.dumps(bench, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in bench.items() if k != "run"}, indent=2, ensure_ascii=False))
    print(f"rapports -> {out}  ; mesures -> results/{name}")


if __name__ == "__main__":
    main()
