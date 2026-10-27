"""Logique de l'interface, sans Streamlit (testable) : lecture des alertes, décisions de l'analyste, graphe d'un cas.
L'interface ne lit que des dossiers d'alerte (JSON) : elle ne fait jamais partie du chemin de détection."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rapport import ALERTS_DIR, REPO

DECISIONS = REPO / "data" / "neurones" / "decisions.jsonl"  # journal des décisions de l'analyste (hors git)
LABELS = {"valider": "À déclarer", "classer": "Classé sans suite"}


def days() -> list[str]:
    return sorted((p.name for p in ALERTS_DIR.glob("*") if (p / "index.json").exists()), reverse=True)


def cases(day: str) -> list[dict]:
    """Cas de la journée, du plus prioritaire au moins prioritaire."""
    index = json.loads((ALERTS_DIR / day / "index.json").read_text(encoding="utf-8"))
    return sorted(index["cases"], key=lambda c: (-c["priority"], c["rank"]))


def dossier(day: str, case_id: str) -> dict:
    return json.loads((ALERTS_DIR / day / f"{case_id}.json").read_text(encoding="utf-8"))


def record(day: str, case_id: str, decision: str, comment: str = "", path: Path = DECISIONS) -> dict:
    """Ajoute une décision au journal (on n'efface jamais : la dernière décision d'un cas fait foi)."""
    if decision not in LABELS:
        raise ValueError(f"décision inconnue : {decision}")
    entry = {"date": datetime.now().isoformat(timespec="seconds"), "day": day, "case_id": case_id,
             "decision": decision, "comment": comment}  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def decisions(day: str, path: Path = DECISIONS) -> dict[str, dict]:
    """Dernière décision de chaque cas de la journée."""
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["day"] == day:
                out[e["case_id"]] = e
    return out


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def case_dot(f: dict, main: str) -> str:
    """Graphe des comptes du cas (langage DOT) : une flèche par opération signalée, le compte principal en évidence."""
    lines = ["digraph cas {", "rankdir=LR;", 'bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fillcolor="#f0efec", color="#52514e", fontname="Helvetica"];',
             'edge [color="#2a78d6", fontname="Helvetica", fontsize=10];']  # fmt: skip
    for a in f["accounts"]:
        label = f"{a['id']}\\n{a['entity']}\\n{a['bank']} ({a['country']})"
        style = ', fillcolor="#eb6834", fontcolor="#ffffff", penwidth=2' if a["id"] == main else ""
        lines.append(f"{_q(a['id'])} [label={_q(label)}{style}];")
    for o in sorted(f["operations"], key=lambda o: o["index"]):
        label = f"{o['date']}\\n{o['amount_fmt']} {o['currency']}"
        lines.append(f"{_q(o['src'])} -> {_q(o['dst'])} [label={_q(label)}];")
    lines.append("}")
    return "\n".join(lines)
