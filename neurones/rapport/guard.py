"""Garde-fou numérique : tout nombre d'un rapport doit figurer dans le dossier de l'alerte.

Un « nombre » est une suite de chiffres isolée (pas collée à une lettre, comme dans l'identifiant 80EB1E930),
avec séparateurs de milliers et décimale à la française (« 62 956,81 »). Les identifiants numériques des comptes
(« 0139342 ») et les dates (« 09/09/2022 05:18 ») sont découpés de la même façon dans le dossier et dans le rapport.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![\w.,])\d+(?:[  ]\d{3})*(?:[.,]\d+)?(?!\w)")


def _norm(token: str) -> str:
    return re.sub(r"[\s  ]", "", token).replace(".", ",")


def numbers(text: str) -> list[str]:
    return [_norm(m) for m in NUMBER.findall(text)]


def allowed(f: object) -> set[str]:
    """Nombres présents dans le dossier : dans ses textes, et ses entiers tels quels."""
    out: set[str] = set()
    if isinstance(f, dict):
        for v in f.values():
            out |= allowed(v)
    elif isinstance(f, list):
        for v in f:
            out |= allowed(v)
    elif isinstance(f, str):
        out.update(numbers(f))
    elif isinstance(f, int) and not isinstance(f, bool):
        out.add(str(f))
    return out


def unknown(text: str, f: dict) -> list[str]:
    """Nombres du texte absents du dossier ; liste vide = texte accepté."""
    ok = allowed(f)
    return [n for n in numbers(text) if n not in ok]
