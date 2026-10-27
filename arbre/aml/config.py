"""Source unique des chemins, vocabulaires, découpages temporels et paramètres du moteur."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # arbre/ : code, résultats et modèles de la solution par arbres
REPO = ROOT.parent  # racine du dépôt : ressources partagées entre solutions (data/, models/, tools/, .venv/)
DATA = REPO / "data"
DATA_RAW = DATA / "raw"
DATA_PROC = DATA / "processed"
DATA_FEAT = DATA / "features"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
BUILD = ROOT / "build"
OUT = ROOT / "out"

DEFAULT_VARIANT = "HI-Small"
SEED = 42
TRAIN_THREADS = 8  # threads fixés pour l'entraînement (reproductibilité)

# Vocabulaires figés : l'ordre définit les codes uint8 transmis au moteur.
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = [
    "US Dollar", "Euro", "Yuan", "Swiss Franc", "UK Pound", "Rupee", "Shekel", "Yen", "Ruble",
    "Canadian Dollar", "Bitcoin", "Australian Dollar", "Mexican Peso", "Saudi Riyal", "Brazil Real",
]  # fmt: skip
TYPOLOGIES = [
    "FAN-OUT", "FAN-IN", "CYCLE", "GATHER-SCATTER", "SCATTER-GATHER", "BIPARTITE", "STACK", "RANDOM",
]  # fmt: skip
ENTITY_TYPES = ["Corporation", "Partnership", "Sole Proprietorship", "Individual", "Country", "Direct"]

# Libellés français des typologies (rapports, alertes)
TYPOLOGY_FR = {
    "FAN-OUT": "Éventail sortant (fan-out)",
    "FAN-IN": "Éventail entrant (fan-in)",
    "CYCLE": "Circuit fermé (cycle)",
    "GATHER-SCATTER": "Collecte puis dispersion (gather-scatter)",
    "SCATTER-GATHER": "Dispersion puis collecte (scatter-gather)",
    "BIPARTITE": "Réseau biparti (bipartite)",
    "STACK": "Empilement en couches (stack)",
    "RANDOM": "Chaîne de passage aléatoire (random)",
    "OTHER": "Hors typologie connue",
}

# Découpage temporel [début, fin) en dates calendaires ; None = jusqu'à la fin du flux.
# Train/val/test = découpage par jours de Multi-GNN (60/20/20) recalculé sur HI-Small ;
# la traîne (11 au 18/09) est évaluée à part, « test_lit » reproduit le test de la littérature.
SPLITS: dict[str, tuple[str, str | None]] = {
    "train": ("2022-09-01", "2022-09-07"),
    "val": ("2022-09-07", "2022-09-09"),
    "test": ("2022-09-09", "2022-09-11"),
    "tail": ("2022-09-11", None),
    "test_lit": ("2022-09-09", None),
}


def minutes_since(t0: datetime, day: str) -> int:
    return int((datetime.fromisoformat(day) - t0).total_seconds() // 60)


@dataclass
class EngineConfig:
    """Paramètres du moteur de features ; ils changent les features, donc entrent dans le hash."""

    win_short: int = 360  # minutes (6 h)
    win_mid: int = 1440  # 24 h
    win_long: int = 5760  # 96 h
    motif_formats: list[str] = field(default_factory=lambda: ["ACH"])
    ring_bits: int = 20  # anneau d'arêtes : 2^20 emplacements
    pair_bits: int = 21  # table des paires : 2^21 emplacements
    pair_ttl: int = 7 * 1440  # éviction des paires inactives (minutes)
    scan_cap: int = 256  # arêtes lues au plus par liste d'adjacence
    fan_cap: int = 64  # voisins examinés pour les motifs à deux sauts
    cycle_max_len: int = 10
    cycle_budget: int = 2000  # expansions de DFS au plus par transaction
    chain_tol: float = 0.10  # écart relatif max pour chaîner un passage de fonds

    def fmt_mask(self) -> int:
        mask = 0
        for f in self.motif_formats:
            mask |= 1 << FORMATS.index(f)
        return mask

    def digest(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
