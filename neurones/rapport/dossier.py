"""Lecture d'un dossier d'alerte (JSON) et vérification du contrat minimal dont le rapport a besoin."""

from __future__ import annotations

import json
from pathlib import Path

# Champs lus par le rapport et l'interface. Tout modèle qui produit ces champs est accepté.
REQUIRED = {
    "case_id": str,
    "priority_pct": str,
    "priority_rank": int,
    "typology": dict,  # code, label, proba_pct
    "period": dict,  # start, end (textes déjà formatés)
    "totals": dict,  # n_ops, n_accounts, usd_fmt
    "accounts": list,  # id, bank, country, entity, entity_accounts, role, n_ops
    "operations": list,  # index, date, src, dst, amount_fmt, currency, usd_fmt, format, score_pct
    "reasons": list,  # text, family, percentile_txt, context
    "max_score_pct": str,
    "threshold_pct": str,
    "model": dict,  # name, method, mode_label
}


class DossierError(ValueError):
    """Dossier qui ne respecte pas le contrat."""


def check(f: dict) -> dict:
    missing = [k for k in REQUIRED if k not in f]
    if missing:
        raise DossierError(f"champs manquants : {missing}")
    bad = [k for k, t in REQUIRED.items() if not isinstance(f[k], t)]
    if bad:
        raise DossierError(f"champs mal typés : {bad}")
    if not f["accounts"] or not f["operations"]:
        raise DossierError("un dossier doit contenir au moins un compte et une opération")
    return f


def load(path: Path) -> dict:
    return check(json.loads(Path(path).read_text(encoding="utf-8")))


def main_account(f: dict) -> dict:
    """Compte au centre du cas : celui qui apparaît dans le plus d'opérations signalées."""
    return max(f["accounts"], key=lambda a: a.get("n_ops", 0))
