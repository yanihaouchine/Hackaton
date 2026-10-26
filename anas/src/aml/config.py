"""Constantes partagées : découpage temporel, seed, correspondances de libellés."""

from __future__ import annotations

from datetime import datetime

SEED = 42

# Découpage temporel [début, fin). La traîne du 11 au 18/09 est évaluée à part.
SPLITS: dict[str, tuple[datetime, datetime]] = {
    "train": (datetime(2022, 9, 1), datetime(2022, 9, 6)),
    "val": (datetime(2022, 9, 6), datetime(2022, 9, 8)),
    "test": (datetime(2022, 9, 8), datetime(2022, 9, 11)),
    "tail": (datetime(2022, 9, 11), datetime(2022, 9, 19)),
}

# Typologies de Patterns.txt -> codes du contrat (étiquettes : évaluation seulement).
PATTERN_TYPOLOGIES = {
    "FAN-OUT": "fan_out",
    "FAN-IN": "fan_in",
    "CYCLE": "cycle",
    "GATHER-SCATTER": "gather_scatter",
    "SCATTER-GATHER": "scatter_gather",
    "BIPARTITE": "bipartite",
    "STACK": "stack",
    "RANDOM": "random",
}

CURRENCY_CODES = {
    "US Dollar": "USD",
    "Euro": "EUR",
    "Yuan": "CNY",
    "Swiss Franc": "CHF",
    "UK Pound": "GBP",
    "Rupee": "INR",
    "Shekel": "ILS",
    "Yen": "JPY",
    "Ruble": "RUB",
    "Bitcoin": "BTC",
    "Canadian Dollar": "CAD",
    "Australian Dollar": "AUD",
    "Mexican Peso": "MXN",
    "Saudi Riyal": "SAR",
    "Brazil Real": "BRL",
}

PAYMENT_FORMATS_FR = {
    "Cheque": "chèque",
    "Credit Card": "carte de crédit",
    "ACH": "virement ACH",
    "Cash": "espèces",
    "Reinvestment": "réinvestissement",
    "Wire": "virement international",
    "Bitcoin": "bitcoin",
}
