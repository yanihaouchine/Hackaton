# Trame de la présentation (15 minutes)

Chaque chiffre affiché vient de `results/*.json`. `python -m aml summarize` produit les tableaux.

## 1. Le problème (1 min)
- LCB-FT sur un flux de transactions : lever des alertes **justifiées**, puis préparer une déclaration de soupçon (Tracfin).
- Contraintes :
  - machine de 4 à 7 W (OrangePi Zero 4, 4 GB) ;
  - flux strictement causal ;
  - chaque alerte doit s'expliquer ;
  - le LLM n'intervient jamais dans la détection.

## 2. Ce que disent les données (2 min)
- L'ACH porte 87 % des illicites. Taux de change fixes, recalculés depuis les données. 38 % des illicites sont hors motifs. La traîne (11 au 18/09) est à 60 % illicite et évaluée à part.
- **Découverte clé : les montants se conservent le long des chaînes** (ratio médian 1,000 sur les relais).
  Un simple détecteur « fonds de passage » atteint 59 % de précision, contre 0,75 % en moyenne sur l'ACH.

## 3. Architecture (2 min)
- Schéma : flux → moteur C (état borné) → forêt compilée → cas → rapport → assistant.
- Un seul code pour l'entraînement et l'inférence : aucun écart possible entre les deux.
- 136 octets par compte, anneau d'arêtes, DFS à budget : coût par transaction borné.

## 4. Idées venues d'autres domaines (2 min)
- Physique : loi des nœuds (Kirchhoff), qui fonde la conservation des flux.
- Forensique blockchain : *taint analysis*, d'où la profondeur de chaîne.
- Vision par ordinateur : cascade de Viola-Jones (mini-modèle de pré-filtrage) ; regroupement de détections en pistes (alertes → cas).
- Traitement du signal : lissage à retard fixe (revue J+1).
- Réseaux : statistiques amorties (EWMA, Welford).
- Systèmes embarqués : arbres compilés en tableaux C.
- Génération de langue : jetons à emplacements, pour un LLM qui n'écrit aucun chiffre.

## 5. Rigueur (2 min)
- Découpage temporel ; seuil fixé sur la validation ; test lu une seule fois par configuration, chaque lecture consignée.
- Tests anti-fuite : invariance par préfixe, futur perturbé, propagation causale, équivalence C/Python sur 180 000 tx, déterminisme bit à bit.
- Ablations par famille ; baselines (règles, features de base).

## 6. Résultats (3 min)
Sources : `results/eval_strict_test-tail-test_lit_20261027-065943.json`, `results/val_*.json`, `results/eval_typology_test_*.json`.
- **Protocole littérature** (test du 09/09 à la fin, traîne comprise) : F1 **76,8** [75,0–78,6] en flux strict, contre 74,9 pour MEGA-GNN et 63–65 pour GFP. Ces méthodes voient le futur.
- **Test propre** (09–10/09) : F1 65,3 [62,4–68,1], précision 0,85 ; précision@100 par jour = 1,0. Traîne évaluée à part : F1 91,2.
- **Tentatives détectées** : 161 sur 174 (FAN-OUT 20/20, SCATTER-GATHER 22/22, CYCLE 21/22, STACK 21/22). Opérations hors motifs : 0,5 % de rappel (limite structurelle).
- **Ablations** : la conservation des flux apporte 2 points de F1 ; les motifs explicites n'apportent rien de plus en score, mais servent de preuves.
- **Typologie nommée** : correcte pour 61 % des cas qui contiennent une opération illicite (193 cas).
- **Étape 2 (propagation des scores, causale)** : test 66,2, protocole littérature **78,8**. C'est la meilleure configuration, et la plus stable (F1 de validation 0,683 ± 0,001).
- **Revue J+1** : aucun gain sur le test propre (65,3). Dans la démo, elle apporte à minuit 33 alertes nouvelles, dont 14 vraies.
- **LI-Small** (2 fois moins d'illicites) : protocole littérature 41,8, au-dessus de GFP, FraudGT et Multi-GNN, en dessous de MEGA-GNN (46,3), qui voit le futur.

## 7. Frugalité (1,5 min)
Sources : `results/bench_ORC_x86_64*.json`, `results/val_pareto_strict.json`.
- **x86, un thread** : 214 000 tx/s sur le flux complet (23,7 s pour 5 M tx), latence p50 de 3,6 µs, état de 166 Mio, forêt de 1,6 Mo, bibliothèque de 43 Ko.
- **Cascade de Viola-Jones** : débit ×33,6 (6 366 → 214 055 tx/s) pour −0,005 de F1.
- **Pareto de la forêt** : 200 arbres donnent un F1 de 0,663 à 15 µs/tx, 1 145 arbres 0,680 à 158 µs/tx.
- **OrangePi** : à mesurer en séance (`scripts/target_setup.sh`, wattmètre, cœurs A55 contre A76).

## 8. Démo (1,5 min)
- Rejeu en direct, puis cas priorisés, rapport de type Tracfin et question à l'assistant.

## 9. Limites et suite (0,5 min)
- Données synthétiques ; illicites hors motifs difficiles ; seuil à adapter à la capacité des analystes.
