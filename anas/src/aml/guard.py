"""Garde-fou : tout nombre d'un texte rédigé doit figurer dans les faits du dossier."""

from __future__ import annotations

import re

from aml.schema import AlertDossier

# Suite de chiffres, avec séparateurs de milliers (espace) et décimales (virgule ou point).
NUMBER = re.compile(r"\d(?:[\d   ]*\d)?(?:[.,]\d+)?")


def _norm(token: str) -> str:
    return re.sub(r"[\s  ]", "", token).replace(".", ",")


def allowed_numbers(dossier: AlertDossier) -> set[str]:
    allowed: set[str] = set()
    for f in dossier.facts:
        allowed.update(_norm(m) for m in NUMBER.findall(f.text))
    return allowed


def unknown_numbers(text: str, dossier: AlertDossier) -> list[str]:
    """Nombres du texte absents des faits ; liste vide = texte accepté."""
    allowed = allowed_numbers(dossier)
    return [m for m in NUMBER.findall(text) if _norm(m) not in allowed]
