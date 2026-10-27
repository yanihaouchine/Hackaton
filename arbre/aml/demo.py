"""Démonstration : rejeu en flux de la période de test, alertes en direct, cas priorisés et rapports.

python -m aml demo --from 2022-09-09 --days 1 [--reports 10] [--llm]
Sorties : out/demo/cases.json, out/demo/<cas>.md (rapport) et out/demo/<cas>.json (dossier de faits).
"""

from __future__ import annotations

import json
import resource
import time
from datetime import datetime, timedelta

import numpy as np

from aml.cases import build_cases
from aml.config import OUT, TYPOLOGY_FR, minutes_since
from aml.data import load_labels
from aml.dossier import History
from aml.report import Reporter
from aml.stream import Monitor


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(variant: str, start: str, days: int, n_reports: int = 10, use_llm: bool = False, model: str = "strict"):
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
    except ImportError:  # affichage simple sans rich
        Console = None

    from aml.typology import TypologyModel

    out = OUT / "demo"
    out.mkdir(parents=True, exist_ok=True)
    mon = Monitor(variant, model)
    s = mon.s
    t0d = datetime.fromisoformat(s.meta["t0"])
    lo = int(np.searchsorted(s.t, minutes_since(t0d, start)))
    end = (datetime.fromisoformat(start) + timedelta(days=days)).date().isoformat()
    hi = int(np.searchsorted(s.t, minutes_since(t0d, end)))
    con = Console() if Console else None
    say = con.print if con else print

    say(f"[bold]Mise en route[/bold] : reconstitution de l'état sur {lo:,} transactions d'historique…" if con
        else f"Mise en route : état reconstitué sur {lo:,} transactions…")  # fmt: skip
    w = mon.warm(lo)
    how = "restauré depuis l'instantané" if mon.restored else f"calculé ({lo / w:,.0f} tx/s), instantané sauvegardé"
    say(f"  état prêt en {w:.1f} s ({how}) — état du moteur {mon.eng.state_bytes() / 2**20:.0f} Mio"
        f"{' — cascade active' if mon.cascade else ''}")  # fmt: skip

    typ = TypologyModel()
    alerts, scores, feats = [], [], []
    n_done, t_engine, t_wall = 0, 0.0, time.perf_counter()
    recent: list[tuple] = []

    def table():
        tb = Table(title=f"Flux du {start} — {n_done:,}/{hi - lo:,} tx — {len(alerts):,} alertes — "
                         f"{n_done / max(t_engine, 1e-9):,.0f} tx/s (moteur) — RSS {_rss_mb():.0f} Mio")  # fmt: skip
        for c in ("Heure", "Émetteur → destinataire", "Montant USD", "Moyen", "Score", "Typologie probable"):
            tb.add_column(c)
        for r in recent[-12:]:
            tb.add_row(*r)
        return tb

    def consume(ch):
        nonlocal n_done, t_engine
        n_done += ch.hi - ch.lo
        t_engine += ch.seconds
        if len(ch.alerts):
            alerts.extend(ch.alerts.tolist())
            scores.extend(ch.scores.tolist())
            feats.append(ch.feats)
            names, _ = typ.predict(ch.feats)
            for i, p, tname in zip(ch.alerts, ch.scores, names, strict=True):
                tt = (t0d + timedelta(minutes=int(s.t[i]))).strftime("%d/%m %H:%M")
                recent.append((tt, f"{int(s.src[i])} → {int(s.dst[i])}", f"{float(s.usd_paid[i]):,.0f}",
                               ["ACH", "Bitcoin", "Cash", "Cheque", "Carte", "Réinvest.", "Wire"][int(s.fmt[i])],
                               f"{p:.2f}", TYPOLOGY_FR.get(tname, tname)))  # fmt: skip

    from aml.model import load_bundle, model_dir
    from aml.stream import Candidates

    j1 = load_bundle("j1") if (model_dir("j1") / "model.json").exists() else None
    review_stats = None
    keep = Candidates([], [], [], []) if j1 is not None else None
    if con:
        with Live(table(), console=con, refresh_per_second=4) as live:
            for ch in mon.replay(hi, chunk=8192, keep=keep):
                consume(ch)
                live.update(table())
    else:
        for ch in mon.replay(hi, chunk=8192, keep=keep):
            consume(ch)
        print(f"{n_done:,} tx, {len(alerts):,} alertes")
    wall = time.perf_counter() - t_wall
    if j1 is not None and days == 1:
        # revue J+1 : à minuit, les candidats du jour sont re-scorés avec ce qui s'est passé depuis (causal)
        t_r = time.perf_counter()
        idx_r, p_r = mon.review(keep, j1)
        al_r = idx_r[p_r >= j1.threshold]
        new = np.setdiff1d(al_r, np.array(alerts, dtype=np.int64))
        review_stats = {"candidates": int(len(idx_r)), "alerts": int(len(al_r)), "new_alerts": int(len(new)),
                        "new_true": int(np.sum(load_labels(variant)["y"][new] == 1)) if len(new) else 0,
                        "seconds": time.perf_counter() - t_r}  # fmt: skip
        say("")
        say(f"Revue J+1 à minuit : {len(al_r):,} alertes sur {len(idx_r):,} candidats, dont {len(new):,} "
            f"nouvelles par rapport au temps réel ({review_stats['seconds']:.1f} s).")  # fmt: skip

    idx = np.array(alerts, dtype=np.int64)
    p = np.array(scores)
    x = np.concatenate(feats) if feats else np.zeros((0, mon.eng.n_features), np.float32)
    cases = build_cases(idx, np.asarray(s.src[idx]), np.asarray(s.dst[idx]), np.asarray(s.t[idx]),
                        np.asarray(s.usd_paid[idx], dtype=np.float64), mon.b.calibrate(p))  # fmt: skip
    say(f"\n{len(alerts):,} alertes regroupées en {len(cases):,} cas en {wall:.1f} s de rejeu.")

    from aml.data import AccountNames

    acc_table = AccountNames(variant)
    hist = History(s, hi)
    rep = Reporter(s, mon.b, hist, acc_table, typology=typ)
    llm = None
    if use_llm:
        from aml.llm import LLM, write_analysis

        llm = LLM()
        say(f"LLM : {llm.kind} {llm.name or ''}")
    pos = {int(a): k for k, a in enumerate(idx)}
    summary = []
    for rank, c in enumerate(cases[:n_reports], start=1):
        rows = [pos[a] for a in c.alerts]
        t_rep = time.perf_counter()
        f = rep.facts(c, x[rows], p[rows], rank=rank)
        if llm is not None and llm.available:
            r = write_analysis(llm, f)
            f["analysis_text"], f["narrative_note"] = r["text"], (
                f"le texte d'analyse a été reformulé par un LLM local ({r['source']}) sous garde : "
                "aucune valeur n'est écrite par le modèle." if r["source"] != "template"
                else "le texte d'analyse est produit par un modèle de phrases (LLM rejeté ou absent).")  # fmt: skip
            f["llm"] = {k: v for k, v in r.items() if k != "text"}
        md = rep.render(f)
        (out / f"{c.case_id}.md").write_text(md)
        (out / f"{c.case_id}.json").write_text(json.dumps(f, ensure_ascii=False, indent=1, default=str))
        summary.append({"case": c.case_id, "priority": c.priority, "typology": f["typology"]["label"],
                        "n_alerts": len(c.alerts), "usd": c.total_usd,
                        "report_seconds": time.perf_counter() - t_rep})  # fmt: skip
        say(f"  #{rank} {c.case_id}  priorité {c.priority:.2f}  {f['typology']['label']}  "
            f"{len(c.alerts)} alerte(s)  {c.total_usd:,.0f} USD")  # fmt: skip
    _write_html(out, summary, start)
    from aml.provenance import write_result

    # chiffres de la démo versionnés (out/ ne l'est pas) ; « new_true » utilise les étiquettes a posteriori,
    # uniquement pour mesurer l'apport de la revue, jamais dans le rejeu
    write_result(f"demo_{start}", {"start": start, "days": days, "n_tx": n_done, "n_alerts": len(alerts),
                                   "n_cases": len(cases), "replay_seconds": wall,
                                   "engine_tx_per_s": n_done / max(t_engine, 1e-9), "cascade": mon.cascade,
                                   "review_j1": review_stats, "top_cases": summary})  # fmt: skip
    (out / "cases.json").write_text(json.dumps({"start": start, "days": days, "n_tx": n_done,
                                                "n_alerts": len(alerts), "n_cases": len(cases),
                                                "replay_seconds": wall, "top": summary}, indent=1))  # fmt: skip
    say(f"Rapports écrits dans {out.relative_to(OUT.parent)}/")


CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e1db;--accent:#8a3b12;--chip:#f1ece4}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#ececec;--muted:#a0a0a5;--line:#2c2c2c;
--accent:#e08a5a;--chip:#232120}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif;
margin:0 auto;max-width:1100px;padding:24px 16px}
h1{font-size:1.5rem}h2{font-size:1.15rem;border-bottom:1px solid var(--line);padding-bottom:4px;margin-top:2rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}
code{background:var(--chip);padding:1px 4px;border-radius:4px;font-size:.85em}
blockquote{border-left:3px solid var(--accent);margin:0;padding:4px 12px;color:var(--muted)}
a{color:var(--accent)}
"""


def _write_html(out, summary: list[dict], start: str) -> None:
    """Pages HTML statiques (index des cas + un rapport par cas) ; ignoré si markdown-it est absent."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return
    from aml.casegraph import case_svg

    md = MarkdownIt("commonmark").enable("table")
    page = (
        "<!doctype html><html lang=fr><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
    )
    for row in summary:
        body = md.render((out / f"{row['case']}.md").read_text())
        facts = json.loads((out / f"{row['case']}.json").read_text())
        svg = case_svg(facts)
        if svg:  # schéma des flux inséré avant la section des opérations
            body = body.replace("<h2>3. Opérations signalées</h2>",
                                f"<h2>Schéma des flux</h2>{svg}<h2>3. Opérations signalées</h2>", 1)  # fmt: skip
        (out / f"{row['case']}.html").write_text(f"{page}<title>{row['case']}</title><style>{CSS}</style>"
                                                 f"<p><a href=index.html>← cas du jour</a></p>{body}")  # fmt: skip
    rows = "".join(
        f"<tr><td><a href={r['case']}.html>{r['case']}</a></td><td>{100 * r['priority']:.1f} %</td>"
        f"<td>{r['typology']}</td><td>{r['n_alerts']}</td><td>{r['usd']:,.0f}</td></tr>".replace(",", "\u202f")
        for r in summary
    )
    (out / "index.html").write_text(
        f"{page}<title>Cas prioritaires</title><style>{CSS}</style><h1>Cas prioritaires — flux du {start}</h1>"
        "<table><tr><th>Cas</th><th>Priorité</th><th>Typologie probable</th><th>Alertes</th>"
        f"<th>Montant (USD)</th></tr>{rows}</table>"
    )
