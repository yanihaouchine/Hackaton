# Sources externes

Toute source externe est consignée ici **avant** son utilisation : URL, licence, date de consultation, usage.

| Source | URL | Licence | Consultée le | Usage |
|---|---|---|---|---|
| IBM Transactions for Anti Money Laundering (AML), variantes HI-Small et LI-Small | https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml | Community Data License Agreement – Sharing 1.0 (vérifiée via l'API Kaggle) | 2026-10-26 | Données d'entraînement et d'évaluation (`scripts/get_data.sh`) |
| Tracfin : « Quand et comment déclarer un soupçon » | https://www.economie.gouv.fr/tracfin/vous-etes-professionnel-declarant/quand-et-comment-declarer-un-soupcon | Information publique (site gouvernemental) | 2026-10-26 | Structure du rapport : identification du déclarant, identification et connaissance du client, éléments d'analyse du soupçon. Aucun contenu repris tel quel |
| Tracfin : guide utilisateur ERMES (version de janvier 2025) | https://ermesdemo.tracfin.finances.gouv.fr/Documentation/GuideUtilisateurs.pdf | Information publique (site gouvernemental) | 2026-10-26 | Ordre des rubriques du rapport (déclarant, contexte, nature du soupçon). Aucun contenu repris tel quel |

Non utilisés : aucune liste GAFI ni aucun taux de change externe. Les taux de change sont **dérivés du jeu de données** (transactions multi-devises de la période d'entraînement, voir `aml/data.py`).
