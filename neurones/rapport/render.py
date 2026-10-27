"""Dossier d'alerte (dict) → rapport Markdown, par template Jinja2, sans LLM."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from rapport.dossier import check, main_account
from rapport.guard import unknown
from rapport.typologies import describe

TEMPLATES = Path(__file__).resolve().parent / "templates"
PRIORITY_RANKS = 10  # au-delà, le cas reste dans la file normale (capacité d'examen du jour)

_env = Environment(loader=FileSystemLoader(TEMPLATES), undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


class GuardError(AssertionError):
    """Le rapport contient un nombre absent du dossier."""


def reason_line(r: dict) -> str:
    """Raison au format de arbre/ : texte, rang parmi les transactions d'apprentissage, contexte éventuel."""
    line = r["text"] + (f" ({r['percentile_txt']})" if r.get("percentile_txt") else "")
    return line + (", *élément de contexte*" if r.get("context") else "")


def render(f: dict) -> str:
    check(f)
    typ_def, typ_why, checks = describe(f["typology"].get("code", "OTHER"))
    urgent = f["priority_rank"] <= PRIORITY_RANKS
    md = _env.get_template("rapport_tracfin.md.j2").render(
        f={"indicators": [], "evidence_lines": [], **f},
        main=main_account(f),
        reason_lines=[reason_line(r) for r in f["reasons"]],
        ops=sorted(f["operations"], key=lambda o: o["index"]),
        typ_def=typ_def,
        typ_why=typ_why,
        checks=checks,
        urgency="Examen prioritaire aujourd'hui : le cas fait partie des plus prioritaires de la journée."
        if urgent
        else "Examen dans la file normale.",
    )
    bad = unknown(md, f)
    if bad:
        raise GuardError(f"nombres absents du dossier : {sorted(set(bad))}")
    return md
