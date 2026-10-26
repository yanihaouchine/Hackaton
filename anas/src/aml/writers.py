"""Rédaction des parties en prose du rapport.

`TemplateWriter` (sans LLM) marche toujours. `LlmWriter` est l'emplacement prévu pour un petit
modèle GGUF local : tant qu'il n'est pas branché, `get_writer("llm")` se replie sur les templates.
"""

from __future__ import annotations

import sys
from typing import Protocol

from aml.guard import unknown_numbers
from aml.schema import AlertDossier
from aml.typologies import REASONS_FR, TYPOLOGIES

SECTIONS = ("synthesis", "analysis")


class Writer(Protocol):
    name: str

    def write(self, dossier: AlertDossier) -> dict[str, str]: ...


def facts_text(dossier: AlertDossier) -> dict[str, str]:
    return {f.key: f.text for f in dossier.facts}


def reason_sentences(dossier: AlertDossier) -> list[str]:
    F = facts_text(dossier)
    out = []
    for i, r in enumerate(dossier.alert.reasons, start=1):
        tpl = REASONS_FR.get(r.code, r.code + " : {value} (seuil : {threshold})")
        out.append(
            tpl.format(
                value=F[f"reason{i}_value"],
                threshold=F.get(f"reason{i}_threshold", "—"),
                window=F.get(f"reason{i}_window", "—"),
            )
        )
    return out


class TemplateWriter:
    name = "template"

    def write(self, dossier: AlertDossier) -> dict[str, str]:
        F = facts_text(dossier)
        typ = TYPOLOGIES[dossier.alert.typology]
        synthesis = (
            f"Le compte {F['subject']} présente un schéma de type « {typ.name} ». "
            f"{F['n_tx']} opérations impliquant {F['n_counterparties']} contreparties distinctes "
            f"ont été relevées en {F.get('span_minutes', '—')}, pour un montant estimé à "
            f"{F.get('total_usd_estimate', '—')}."
        )
        analysis = f"{typ.definition} {typ.why_suspicious}"
        return {"synthesis": synthesis, "analysis": analysis}


class LlmUnavailable(RuntimeError):
    pass


class LlmWriter:
    """Emplacement du LLM (étape à venir). Le LLM recevra les faits et écrira des références [Fx] ;
    le code remplacera les références par les textes des faits, puis le garde-fou validera."""

    name = "llm"

    def __init__(self, model_path: str | None = None) -> None:
        raise LlmUnavailable("rédaction par LLM pas encore branchée")

    def write(self, dossier: AlertDossier) -> dict[str, str]:  # pragma: no cover
        raise NotImplementedError


def get_writer(mode: str = "template", model_path: str | None = None) -> Writer:
    if mode == "llm":
        try:
            return LlmWriter(model_path)
        except LlmUnavailable as e:
            print(f"[rapport] LLM indisponible ({e}) : repli sur les templates", file=sys.stderr)
    return TemplateWriter()


def write_checked(writer: Writer, dossier: AlertDossier) -> tuple[dict[str, str], str]:
    """Applique le garde-fou ; en cas de nombre inconnu, repli sur les templates."""
    sections = writer.write(dossier)
    bad = [n for s in SECTIONS for n in unknown_numbers(sections[s], dossier)]
    if bad and writer.name != "template":
        print(f"[rapport] nombres hors faits {bad} : repli sur les templates", file=sys.stderr)
        return TemplateWriter().write(
            dossier
        ), f"rejeté ({len(bad)} nombre(s) hors faits), repli template"
    if bad:
        raise AssertionError(f"le template produit des nombres hors faits : {bad}")
    return sections, "accepté"
