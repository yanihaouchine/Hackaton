"""Énergie mesurée par RAPL (compteur du paquet CPU Intel) : détection en flux et génération des rapports.

RAPL compte toute la consommation du processeur, y compris ce qui tourne en fond. On mesure donc d'abord la
puissance au repos, puis on retranche « puissance au repos × durée » à chaque mesure : il reste l'énergie
due au travail mesuré. Un seul thread de calcul ; résultat dans results/energie_<machine>.json.
Le capteur n'est lisible qu'en root : `sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj` (temporaire).
Usage (depuis neurones/) : ../.venv/bin/python -m mesures.energie [--date 2022-09-09]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from rapport import ALERTS_DIR, ROOT  # noqa: E402  (rapport ajoute arbre/ au chemin d'import)

RAPL = Path("/sys/class/powercap/intel-rapl:0")
IDLE_S = 20.0
REPORT_PASSES = 200  # 30 rapports × 200 passes : assez long pour que le compteur soit précis


def read_uj() -> int:
    return int((RAPL / "energy_uj").read_text())


def joules(e0: int, e1: int) -> float:
    """Différence de compteur, en tenant compte du retour à zéro."""
    if e1 < e0:
        e1 += int((RAPL / "max_energy_range_uj").read_text())
    return (e1 - e0) / 1e6


def measure(fn) -> tuple[float, float, object]:
    """(joules, secondes, résultat de fn)."""
    e0, t0 = read_uj(), time.perf_counter()
    out = fn()
    return joules(e0, read_uj()), time.perf_counter() - t0, out


def main() -> None:
    from aml.config import minutes_since
    from aml.provenance import run_info
    from aml.stream import Monitor

    from rapport.dossier import load
    from rapport.render import render

    a = argparse.ArgumentParser()
    a.add_argument("--date", default="2022-09-09")
    args = a.parse_args()
    try:
        read_uj()
    except PermissionError:
        raise SystemExit(f"RAPL illisible : lancer `sudo chmod o+r {RAPL}/energy_uj`") from None

    # 1. Repos
    j_idle, s_idle, _ = measure(lambda: time.sleep(IDLE_S))
    p_idle = j_idle / s_idle

    # 2. Détection en flux sur une journée (forêt strict + cascade de arbre/, lecture seule)
    mon = Monitor("HI-Small", "strict", cascade=True)
    s = mon.s
    t0 = datetime.fromisoformat(s.meta["t0"])
    end = (datetime.fromisoformat(args.date) + timedelta(days=1)).date().isoformat()
    lo = int(np.searchsorted(s.t, minutes_since(t0, args.date)))
    hi = int(np.searchsorted(s.t, minutes_since(t0, end)))
    mon.warm(lo, use_snapshot=False)
    j_det, s_det, n_alerts = measure(lambda: sum(len(ch.alerts) for ch in mon.replay(hi, chunk=8192)))
    n_tx = hi - lo

    # 3. Rapports sans LLM (dossiers produits par rapport.make_alerts)
    dossiers = [load(p) for p in sorted((ALERTS_DIR / args.date).glob("C*.json"))]
    if not dossiers:
        raise SystemExit("aucun dossier : lancer d'abord `python -m rapport.make_alerts`")
    render(dossiers[0])  # compilation du template hors mesure
    j_rep, s_rep, _ = measure(lambda: [render(f) for _ in range(REPORT_PASSES) for f in dossiers])
    n_rep = REPORT_PASSES * len(dossiers)

    def net(j: float, sec: float) -> float:
        return max(j - p_idle * sec, 0.0)

    out = {
        "run": run_info(),
        "machine": platform.node(),
        "method": "RAPL, paquet CPU (package-0) ; énergie nette = brute − puissance au repos × durée ; 1 thread",
        "idle": {"seconds": round(s_idle, 1), "watts": round(p_idle, 2)},
        "detection": {
            "date": args.date, "n_tx": n_tx, "n_alerts": int(n_alerts), "seconds": round(s_det, 2),
            "tx_per_s": round(n_tx / s_det), "joules_brut": round(j_det, 2), "joules_net": round(net(j_det, s_det), 2),
            "watts_moyens": round(j_det / s_det, 2), "uj_net_par_tx": round(1e6 * net(j_det, s_det) / n_tx, 3),
        },
        "rapports_sans_llm": {
            "n_rapports": n_rep, "seconds": round(s_rep, 2), "joules_brut": round(j_rep, 2),
            "joules_net": round(net(j_rep, s_rep), 2), "watts_moyens": round(j_rep / s_rep, 2),
            "mj_net_par_rapport": round(1e3 * net(j_rep, s_rep) / n_rep, 4),
        },
        "rapports_avec_llm": "non mesuré : le LLM tourne sur le GPU de DeepEirb, hors de portée de RAPL",
    }  # fmt: skip
    path = ROOT / "results" / f"energie_{platform.node()}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=2, ensure_ascii=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
