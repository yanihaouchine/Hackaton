"""Dossiers d'alerte pour la démo, produits par le mélange figé forêt + modèle séquentiel, au format de arbre/.

Rejoue une tranche d'un jour de TEST (le jury choisit le jour et l'heure) : chaque transaction est notée par le
mélange figé (poids et seuil de results/val_blend_hist.json), à partir des features causales du moteur et de
l'historique strictement antérieur des comptes. Les alertes sont regroupées en cas (aml.cases), puis chaque cas
reçoit un dossier de arbre/ (aml.report.Reporter : comptes, opérations, typologie, raisons TreeSHAP de la forêt),
complété par les transactions passées décisives selon le modèle séquentiel.
Deux processus, car torch et lightgbm (TreeSHAP) ne peuvent pas être chargés ensemble sur macOS :
  1. --phase score (torch, sans lightgbm) : scores, alertes, raisons d'historique → _score.npz et _hist.json ;
  2. dossiers (lightgbm, sans torch) : Reporter, raisons complétées, index.json.
La vérité (étiquettes) n'est lue qu'à la fin, pour truth.json, que l'interface n'affiche pas.
Usage (depuis neurones/) : python -m mlp.make_alerts_hist --date 2022-09-09 --start 08:00 --hours 3
Sorties : data/neurones/alertes/<date>_<HHhMM>_melange/ (lue par l'interface d'Anas comme une journée)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta

import numpy as np

from mlp.common import REPO, RESULTS, load_stream

ALERTS_DIR = REPO / "data" / "neurones" / "alertes"
MODEL = "hist_h64-32_g16_s42"
TEST_DAYS = ("2022-09-09", "2022-09-10")


def window(stream, date: str, start: str, hours: float) -> tuple[int, int]:
    from aml.config import minutes_since

    t0 = datetime.fromisoformat(stream.meta["t0"])
    a = datetime.fromisoformat(f"{date}T{start}")
    b = min(a + timedelta(hours=hours), datetime.fromisoformat(date) + timedelta(days=1))
    lo = int(np.searchsorted(stream.t, minutes_since(t0, a.isoformat())))
    hi = int(np.searchsorted(stream.t, minutes_since(t0, b.isoformat())))
    return lo, hi


def out_dir(date: str, start: str) -> object:
    return ALERTS_DIR / f"{date}_{start.replace(':', 'h')}_melange"


def phase_score(date: str, start: str, hours: float, n_cases: int) -> None:
    """Processus torch : scores du mélange, alertes, cas et raisons d'historique des cas prioritaires."""
    from aml.cases import build_cases
    from aml.data import AccountNames
    from aml.forest import c_raw
    from aml.model import load_bundle

    from mlp.explain_hist import contributions, describe_event
    from mlp.history import batch
    from mlp.train_hist import HistScorer

    t_start = time.perf_counter()
    frozen = json.loads((RESULTS / "val_blend_hist.json").read_text())
    w, thr = frozen["full_val"]["w"], frozen["full_val"]["threshold"]
    sc = HistScorer(frozen["model"])
    s = sc.stream
    lo, hi = window(s, date, start, hours)
    rows = np.arange(lo, hi)
    forest = load_bundle("strict").forest
    t = time.perf_counter()
    z = w * c_raw(forest, np.asarray(sc.X[lo:hi], dtype=np.float32)).astype(np.float64) + (1 - w) * sc.logit(rows)
    score_s = time.perf_counter() - t
    hit = np.flatnonzero(z >= thr)
    idx, p = rows[hit], 1.0 / (1.0 + np.exp(-z[hit]))
    cases = build_cases(idx, np.asarray(s.src[idx]), np.asarray(s.dst[idx]), np.asarray(s.t[idx]),
                        np.asarray(s.usd_paid[idx], dtype=np.float64), p)  # fmt: skip
    pos = {int(i): k for k, i in enumerate(idx)}
    names = AccountNames("HI-Small")
    hist = {}
    for c in cases[:n_cases]:
        top = max(c.alerts, key=lambda i: p[pos[i]])
        r = np.array([top])
        x = np.ascontiguousarray(sc.X[r][:, sc.cols], dtype=np.float32)
        es, ms, ed, md = batch(*sc.hist, r, sc.ram)
        _, cs, cd = contributions(sc.net, x, es, ms, ed, md)
        past = [(float(cs[0, j]), "src", int(sc.hist[0][top, j])) for j in range(cs.shape[1]) if ms[0, j]]
        past += [(float(cd[0, j]), "dst", int(sc.hist[1][top, j])) for j in range(cd.shape[1]) if md[0, j]]
        hist[c.case_id] = [{"text": describe_event(sc, names, top, i, side), "contribution": cval}
                           for cval, side, i in sorted(past, reverse=True)[:3] if cval > 0]  # fmt: skip
    d = out_dir(date, start)
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "_score.npz", idx=idx, p=p, lo=lo, hi=hi)
    meta = {"w": w, "threshold": thr, "model": frozen["model"], "n_tx": int(hi - lo), "score_seconds": score_s,
            "phase_seconds": round(time.perf_counter() - t_start, 1)}  # fmt: skip
    (d / "_hist.json").write_text(json.dumps({"meta": meta, "history": hist}, ensure_ascii=False, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--date", default="2022-09-09", choices=TEST_DAYS, help="jour de TEST uniquement")
    p.add_argument("--start", default="00:00", help="heure de début (HH:MM)")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--cases", type=int, default=30, help="nombre de dossiers écrits (cas les plus prioritaires)")
    p.add_argument("--phase", choices=["score"], help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.phase == "score":
        phase_score(a.date, a.start, a.hours, a.cases)
        return

    t_start = time.perf_counter()
    cmd = [sys.executable, "-m", "mlp.make_alerts_hist", "--phase", "score", "--date", a.date, "--start", a.start,
           "--hours", str(a.hours), "--cases", str(a.cases)]  # fmt: skip
    subprocess.run(cmd, check=True)

    from aml.cases import build_cases
    from aml.data import AccountNames, load_labels
    from aml.dossier import History
    from aml.model import load_bundle
    from aml.report import Reporter, narrative
    from aml.typology import TypologyModel

    d = out_dir(a.date, a.start)
    sc = np.load(d / "_score.npz")
    idx, p, lo, hi = sc["idx"], sc["p"], int(sc["lo"]), int(sc["hi"])
    side = json.loads((d / "_hist.json").read_text())
    stream, _ = load_stream()
    cases = build_cases(idx, np.asarray(stream.src[idx]), np.asarray(stream.dst[idx]), np.asarray(stream.t[idx]),
                        np.asarray(stream.usd_paid[idx], dtype=np.float64), p)  # fmt: skip
    from mlp.common import load_features

    X, _, _ = load_features()
    bundle = load_bundle("strict")
    rep = Reporter(stream, bundle, History(stream, hi), AccountNames("HI-Small"), typology=TypologyModel())
    pos = {int(i): k for k, i in enumerate(idx)}
    index = []
    for rank, c in enumerate(cases[: a.cases], start=1):
        rows = [pos[i] for i in c.alerts]
        f = rep.facts(c, np.asarray(X[np.asarray(c.alerts)], dtype=np.float32), p[rows], rank=rank)
        # Raisons : forêt (TreeSHAP) puis historique selon le modèle séquentiel ; contexte en dernier
        main_r = [r for r in f["reasons"] if not r.get("context")]
        ctx = [r for r in f["reasons"] if r.get("context")]
        hist = [{"feature": "historique", "family": "historique des comptes", "text": h["text"],
                 "contribution": h["contribution"], "percentile": float("nan"), "percentile_txt": "",
                 "context": False} for h in side["history"].get(c.case_id, [])]  # fmt: skip
        f["reasons"] = main_r + hist + ctx
        # Scores : probabilités du mélange (la calibration de la forêt ne s'applique pas)
        for o in f["operations"]:
            o["score_pct"] = f"{100 * float(p[pos[o['index']]]):.1f} %".replace(".", ",") if o["index"] in pos else ""
        top = float(p[rows].max())
        f["max_score_pct"] = "> 99,9 %" if top > 0.999 else f"{100 * top:.1f} %".replace(".", ",")
        f["threshold_pct"] = f"{100 / (1 + np.exp(-side['meta']['threshold'])):.1f} %".replace(".", ",")
        f["model"] = {"name": f"mélange forêt strict + séquentiel ({side['meta']['model']})",
                      "method": "TreeSHAP (forêt) et gradient × entrée sur l'historique (réseau séquentiel)",
                      "mode": "strict", "mode_label": "temps réel (score à l'arrivée)"}  # fmt: skip
        f["detector"] = {"solution": "neurones/", "model": "mélange forêt + séquentiel", "w": side["meta"]["w"]}
        f["analysis_text"] = narrative(f)
        (d / f"{c.case_id}.json").write_text(json.dumps(f, ensure_ascii=False, indent=1, default=str))
        index.append({"case_id": c.case_id, "rank": rank, "priority": c.priority, "priority_pct": f["priority_pct"],
                      "typology": f["typology"]["label"], "n_alerts": len(c.alerts),
                      "usd_fmt": f["totals"]["usd_fmt"], "main_account": f["accounts"][0]["id"],
                      "period_start": f["period"]["start"]})  # fmt: skip
    m = side["meta"]
    summary = {"date": a.date, "start": a.start, "hours": a.hours, "n_tx": m["n_tx"], "n_alerts": int(len(idx)),
               "n_cases": len(cases), "n_dossiers": len(index), "score_seconds": round(m["score_seconds"], 2),
               "score_us_per_tx": round(1e6 * m["score_seconds"] / max(m["n_tx"], 1), 1),
               "total_seconds": round(time.perf_counter() - t_start, 1), "cases": index}  # fmt: skip
    (d / "index.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    # Vérité de la tranche (étiquettes lues seulement maintenant ; l'interface ne lit pas ce fichier)
    lab = load_labels("HI-Small")
    y = lab["y"][lo:hi].astype(bool)
    alert = np.zeros(hi - lo, bool)
    alert[idx - lo] = True
    att = lab["attempt"][lo:hi]
    ids = np.unique(att[att >= 0])
    truth = {"n_illicit": int(y.sum()), "caught": int((y & alert).sum()), "missed": int((y & ~alert).sum()),
             "false_alerts": int((~y & alert).sum()), "missed_non_pattern": int((y & ~alert & (att < 0)).sum()),
             "attempts": int(len(ids)), "attempts_detected": int(sum(alert[att == k].any() for k in ids)),
             "true_cases": int(sum(bool(lab["y"][np.asarray(c.alerts)].any()) for c in cases))}  # fmt: skip
    (d / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2, ensure_ascii=False))
    print("vérité :", json.dumps(truth, ensure_ascii=False))
    print(f"dossiers -> {d}")


if __name__ == "__main__":
    main()
