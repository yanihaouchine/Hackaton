"""Dossier d'alerte (dict) → rapport Markdown, par template Jinja2. Un LLM local peut reformuler le résumé
(`rapport/llm.py`), sous garde ; sans LLM, ou s'il est rejeté, le résumé du template est utilisé."""

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


def summary(f: dict) -> str:
    """Résumé rédigé par template : la référence, et le texte que le LLM peut reformuler."""
    return (
        f"Entre le {f['period']['start']} et le {f['period']['end']}, {f['totals']['n_accounts']} comptes ont réalisé "
        f"{f['totals']['n_ops']} opérations signalées, pour un total de **{f['totals']['usd_fmt']}** "
        f"(contre-valeur en dollars). Le cas est classé au rang **{f['priority_rank']}** de la journée "
        f"(priorité {f['priority_pct']}) et se rapproche de la typologie « {f['typology']['label']} »."
    )


def render(f: dict, writer=None) -> str:
    """`writer` : client LLM (rapport/llm.py) ou None pour le template seul."""
    check(f)
    text, redaction = summary(f), "rapport rédigé par template, sans LLM"
    meta = {}
    if writer is not None:
        from rapport.llm import rewrite

        text, status = rewrite(f, writer)
        meta = {"redaction": status}  # le nom du modèle peut contenir des chiffres : il fait partie des faits
        redaction = (
            f"résumé reformulé par un LLM local ({status['model']}) sous garde, le reste par template"
            if status["source"] == "llm"
            else f"rapport rédigé par template (LLM non retenu : {status['reason']})"
        )
    typ_def, typ_why, checks = describe(f["typology"].get("code", "OTHER"))
    urgent = f["priority_rank"] <= PRIORITY_RANKS
    md = _env.get_template("rapport_tracfin.md.j2").render(
        f={"indicators": [], "evidence_lines": [], **f},
        main=main_account(f),
        summary=text,
        redaction=redaction,
        reason_lines=[reason_line(r) for r in f["reasons"]],
        ops=sorted(f["operations"], key=lambda o: o["index"]),
        typ_def=typ_def,
        typ_why=typ_why,
        checks=checks,
        urgency="Examen prioritaire aujourd'hui : le cas fait partie des plus prioritaires de la journée."
        if urgent
        else "Examen dans la file normale.",
    )
    bad = unknown(md, {**f, **meta})
    if bad:
        raise GuardError(f"nombres absents du dossier : {sorted(set(bad))}")
    return md
