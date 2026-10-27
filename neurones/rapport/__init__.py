"""Rapport inspiré de la déclaration de soupçon TRACFIN, à partir de dossiers d'alerte (JSON).

Le rapport ne dépend d'aucun modèle : il lit un dossier au format produit par `arbre/` (`Reporter.facts`),
quel que soit le modèle qui a levé l'alerte (forêt, MLP ou combinaison), tant qu'il fournit les mêmes champs.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # neurones/
REPO = ROOT.parent
ARBRE = REPO / "arbre"
ALERTS_DIR = REPO / "data" / "neurones" / "alertes"  # dossiers générés, régénérables (hors git)

if str(ARBRE) not in sys.path:
    sys.path.insert(0, str(ARBRE))
