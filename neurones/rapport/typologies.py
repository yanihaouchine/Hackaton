"""Définition de chaque typologie et vérifications proposées à l'analyste.
Textes sans aucun chiffre : tous les nombres du rapport viennent du dossier."""

from __future__ import annotations

TYPOLOGIES = {
    "FAN-OUT": (
        "Un compte envoie des fonds vers de nombreux bénéficiaires distincts sur une courte période.",
        "Éclater une somme entre plusieurs comptes complique le suivi des fonds : c'est une étape classique "
        "d'empilement.",
        ["Lien économique entre le titulaire et les bénéficiaires.", "Origine des fonds dispersés."],
    ),
    "FAN-IN": (
        "Un compte reçoit des fonds de nombreux émetteurs distincts sur une courte période.",
        "Rassembler de nombreux versements sur un compte peut servir à regrouper des fonds d'origine illicite.",
        ["Nature de l'activité du titulaire : la collecte est-elle légitime ?", "Destination des fonds rassemblés."],
    ),
    "CYCLE": (
        "Des fonds partent d'un compte, traversent plusieurs comptes et reviennent à leur point de départ.",
        "Un aller-retour sans raison économique sert à créer un historique artificiel ou à justifier faussement "
        "l'origine des fonds.",
        ["Justification économique de chaque étape du circuit."],
    ),
    "GATHER-SCATTER": (
        "Un compte reçoit des fonds de plusieurs émetteurs puis les redistribue vers plusieurs bénéficiaires.",
        "Le compte sert de plaque tournante : les fonds n'y restent pas, ils transitent.",
        ["Rôle réel du titulaire : compte de passage ?", "Durée de séjour des fonds sur le compte."],
    ),
    "SCATTER-GATHER": (
        "Des fonds sont dispersés vers des intermédiaires puis rassemblés sur un même compte.",
        "Le passage par des intermédiaires masque le lien entre l'émetteur et le bénéficiaire final.",
        ["Liens entre les intermédiaires.", "Identité du bénéficiaire final."],
    ),
    "BIPARTITE": (
        "Un groupe de comptes émetteurs paie un groupe de comptes bénéficiaires, de façon croisée.",
        "Des flux croisés entre deux groupes de comptes brouillent l'origine et la destination des fonds.",
        ["Liens entre les comptes de chaque groupe."],
    ),
    "STACK": (
        "Des fonds traversent plusieurs couches successives de comptes.",
        "L'empilement en couches allonge la piste pour masquer l'origine des fonds.",
        ["Origine des fonds en tête de chaîne.", "Bénéficiaire final."],
    ),
    "RANDOM": (
        "Des fonds passent de compte en compte selon un trajet sans logique apparente.",
        "Un trajet sans logique économique vise à brouiller la piste des fonds.",
        ["Origine des fonds en tête de chaîne.", "Bénéficiaire final."],
    ),
}
OTHER = (
    "La combinaison d'indicateurs ne correspond à aucune typologie connue.",
    "Plusieurs indicateurs de risque sont réunis sur les mêmes comptes.",
    ["Examen manuel de l'ensemble des opérations."],
)


def describe(code: str) -> tuple[str, str, list[str]]:
    """(définition, pourquoi c'est suspect, vérifications à mener)."""
    return TYPOLOGIES.get(code, OTHER)
