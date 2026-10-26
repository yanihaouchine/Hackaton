"""Dictionnaire des typologies et des raisons, en français, pour les rapports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Typology:
    name: str
    kickoff_name: str | None
    definition: str
    why_suspicious: str
    checks: tuple[str, ...]


TYPOLOGIES = {
    "fan_out": Typology(
        "Dispersion (fan-out)",
        "dispersion",
        "Un compte envoie des fonds vers de nombreux bénéficiaires distincts sur une courte période.",
        "Fractionner une somme entre plusieurs comptes complique le suivi des fonds : "
        "c'est une étape classique d'empilement (layering).",
        (
            "Lien économique entre le titulaire et les bénéficiaires.",
            "Origine des fonds dispersés.",
        ),
    ),
    "fan_in": Typology(
        "Rassemblement (fan-in)",
        "rassemblement",
        "Un compte reçoit des fonds de nombreux émetteurs distincts sur une courte période.",
        "La collecte de nombreux petits versements sur un compte peut servir à regrouper "
        "des fonds d'origine illicite avant leur réutilisation.",
        (
            "Nature de l'activité du titulaire (collecte légitime ?).",
            "Destination des fonds rassemblés.",
        ),
    ),
    "gather_scatter": Typology(
        "Rassemblement puis dispersion (gather-scatter)",
        "rassemblement puis dispersion",
        "Un compte reçoit des fonds de nombreux émetteurs puis les redistribue vers de nombreux bénéficiaires.",
        "Le compte joue le rôle de plaque tournante : les fonds n'y restent pas, ils transitent.",
        ("Rôle du titulaire (compte de passage ?).", "Durée de séjour des fonds sur le compte."),
    ),
    "scatter_gather": Typology(
        "Dispersion puis rassemblement (scatter-gather)",
        None,
        "Des fonds sont dispersés vers des intermédiaires puis rassemblés sur un même compte.",
        "Le passage par des intermédiaires masque le lien entre l'émetteur et le bénéficiaire final.",
        ("Liens entre les intermédiaires.",),
    ),
    "cycle": Typology(
        "Circuit fermé (cycle)",
        "circuit fermé",
        "Des fonds partent d'un compte, traversent plusieurs comptes et reviennent au compte d'origine.",
        "Un aller-retour sans raison économique sert à créer un historique artificiel "
        "ou à justifier faussement l'origine des fonds.",
        ("Justification économique de chaque étape du circuit.",),
    ),
    "rapid_chain": Typology(
        "Chaîne rapide",
        "chaîne rapide",
        "Des fonds reçus sont réexpédiés presque en totalité, de compte en compte, en moins de 24 heures.",
        "Des fonds qui traversent plusieurs comptes sans y rester sont typiques de l'empilement "
        "destiné à brouiller leur origine.",
        ("Origine des fonds en tête de chaîne.", "Bénéficiaire final."),
    ),
    "structuring": Typology(
        "Fractionnement",
        "fractionnement",
        "Plusieurs opérations d'un montant juste inférieur à un seuil déclaratif, sur une courte période.",
        "Rester sous un seuil de déclaration ou de contrôle évite la vigilance automatique : "
        "c'est le schéma dit de « smurfing ».",
        ("Cohérence des montants avec l'activité habituelle du client.",),
    ),
    "dormant": Typology(
        "Compte réveillé",
        "compte réveillé",
        "Un compte resté inactif plusieurs jours effectue soudain une opération importante.",
        "Un compte dormant réactivé peut être utilisé comme compte de passage, voire être contrôlé par un tiers.",
        ("Changement récent de titulaire ou de mandataire.", "Origine des fonds."),
    ),
}
for _code in ("bipartite", "stack", "random", "unknown"):
    TYPOLOGIES[_code] = Typology(
        "Schéma non caractérisé",
        None,
        "Combinaison d'indicateurs ne correspondant pas à une typologie nommée.",
        "Plusieurs indicateurs de risque sont réunis sur le même compte.",
        ("Examen manuel de l'ensemble des opérations.",),
    )

# Phrases par code de raison. {value}, {threshold}, {window} : textes de faits injectés par le code.
REASONS_FR = {
    "FANOUT_DISTINCT_DEST": "fonds envoyés vers {value} bénéficiaires distincts en {window} (seuil : {threshold})",
    "FANIN_DISTINCT_SRC": "fonds reçus de {value} émetteurs distincts en {window} (seuil : {threshold})",
    "RAPID_CHAIN_DEPTH": "fonds ayant traversé {value} virements successifs en moins de {window} (seuil : {threshold})",
    "CYCLE_CLOSED": "fonds revenus au compte d'origine après {value} virements (seuil : {threshold})",
    "STRUCTURING_BELOW_LIMIT": "{value} opérations d'un montant juste inférieur au seuil de contrôle en {window} (seuil : {threshold})",
    "DORMANT_REACTIVATED": "opération importante après {value} heures d'inactivité (seuil : {threshold})",
}
