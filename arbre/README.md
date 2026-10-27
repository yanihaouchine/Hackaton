# Détection LCB-FT en flux, frugale et explicable

À partir du flux de transactions IBM AML (variante HI-Small), le système :

1. lève des alertes **en temps réel**, chaque transaction étant scorée à son arrivée avec les seules transactions antérieures ;
2. regroupe les alertes en **cas priorisés**, chacun avec une typologie nommée ;
3. produit pour chaque cas un **projet de déclaration de soupçon** de type TRACFIN, qui justifie l'alerte ;
4. répond aux questions de l'analyste à partir du seul dossier du cas.

La détection est 100 % algorithmique : un moteur de features en C, puis une forêt d'arbres. Un LLM local, optionnel et sous garde, ne fait que reformuler le texte. Il n'écrit jamais un chiffre.

Les résultats chiffrés sont dans [`results/`](results/) : chaque fichier JSON porte la date, le commit, la seed et la machine. Ce README ne cite que ces fichiers.

## Démarrage rapide

Le dépôt accueille plusieurs solutions ; celle-ci (forêt d'arbres) est dans `arbre/`. Les ressources partagées restent
à la racine du dépôt : `data/` (jeu IBM AML et fichiers dérivés), `models/` (GGUF), `.venv/` et `scripts/get_data*.sh`.
Les commandes ci-dessous se lancent depuis `arbre/`.

### Deux phases, deux machines
La frugalité revendiquée concerne **l'inférence** : le rejeu du flux, le scoring, les cas et les rapports. La préparation des données et l'entraînement sont une phase **hors ligne**, plus gourmande, menée une fois sur un poste de développement ; ses produits (flux préparé, modèles, instantané d'état) sont ensuite copiés sur la cible.

| | Phase hors ligne (poste de développement) | Phase d'inférence (cible frugale) |
|---|---|---|
| Rôle | préparer le flux, calculer les features, entraîner, choisir les seuils, évaluer | rejouer le flux, scorer, regrouper en cas, rédiger les rapports, répondre à l'analyste |
| Commandes | `prepare`, `features`, `rules`, `ablate`, `train`, `cascade`, `typology`, `evaluate`, `explain-eval`, `summarize`, `pytest` | `bench`, `demo`, `ask` |
| Dépendances | numpy, polars, lightgbm, scikit-learn (extra `train`) | numpy et le moteur C ; jinja2, rich, markdown-it-py pour les rapports et la démo ; lightgbm en option (TreeSHAP) |
| Mémoire | charge le CSV complet en RAM (tri, encodage) : prévoir plusieurs Go | état borné du moteur (voir « Frugalité ») |
| Matériel | CPU x86-64 ou ARM64 ; **aucun GPU requis** (LightGBM s'entraîne sur CPU) | OrangePi Zero 4, Jetson ou tout CPU x86-64/ARM64 |

Sur un processeur x86 **sans AVX2** (Celeron, Atom, Pentium d'entrée de gamme), la wheel standard de polars plante à l'import (« Instruction non permise ») : installer `polars[rtcompat]` à la place. La phase d'inférence n'utilise pas polars et n'est pas concernée.

### Phase hors ligne (poste de développement)
```bash
../scripts/get_data.sh HI-Small              # données Kaggle (publiques), dans ../data/raw/
python -m aml prepare                        # CSV → flux trié et encodé (≈ 15 s)
python -m aml features --hindsight           # moteur C sur tout le flux + features J+1
python -m aml rules                          # baseline B0 (règles expertes)
python -m aml ablate                         # ablations par famille (validation)
python -m aml train                          # modèle strict, 5 seeds, seuil fixé sur la validation
python -m aml cascade                        # mini-modèle de pré-filtrage (seuil fixé sur la validation)
python -m aml typology                       # classifieur de typologie
python -m aml train --name j1 --hindsight    # modèle « revue J+1 »
python -m aml evaluate --model strict --splits test,tail,test_lit   # test : lu une seule fois
python -m aml explain-eval                   # fidélité et couverture des explications
python -m aml summarize                      # tableaux Markdown à partir de results/
pytest                                       # causalité, équivalence C/Python, forêt, garde LLM, portabilité aarch64
```

### Phase d'inférence (frugale)
Sur la cible ou sur le poste de développement, à partir des produits de la phase hors ligne :
```bash
python -m aml bench                          # frugalité (débit, latences, RAM, tailles, énergie)
python -m aml demo --from 2022-09-09         # rejeu en direct + cas + rapports (out/demo/index.html)
python -m aml ask C08-0001 "Pourquoi ce cas ?"   # assistant d'analyse
```

### Sur la carte cible (OrangePi Zero 4, Jetson)
Aucun entraînement sur la carte : on copie le code, le flux préparé (≈ 290 Mo), les modèles (≈ 12 Mo) et, en option, l'instantané d'état (≈ 170 Mo), qui évite de rejouer l'historique.

```bash
cd .. && rsync -aR --exclude out --exclude build \
      arbre data/processed/HI-Small data/state cible:Hackathon/   # depuis la racine du dépôt, sur le poste de développement
ssh cible 'cd Hackathon/arbre && scripts/target_setup.sh venv && scripts/target_setup.sh build'
ssh cible 'cd Hackathon/arbre && scripts/target_setup.sh bench'   # wattmètre : repos 60 s, puis pendant le rejeu
ssh cible 'cd Hackathon/arbre && .venv-target/bin/python -m aml bench --watts-idle <W> --watts-load <W> --tag watt'
ssh cible 'cd Hackathon/arbre && scripts/target_setup.sh demo'
```

`scripts/target_setup.sh` installe numpy, jinja2, rich et markdown-it-py, plus lightgbm en option : sans lui, les explications passent par Saabas au lieu de TreeSHAP. Il compile ensuite le moteur avec le gcc de la carte, sans `-march`, et mesure aussi le cœur le plus lent et le plus rapide (`taskset`).

## Architecture

```
CSV ─prepare─► flux encodé (ids, USD, KYC ; étiquettes à part)
    ─► moteur C (état borné, 104 features causales, forêt compilée) ─► score à l'arrivée ─► alertes
    ─► cas (union-find sur les comptes, 96 h) ─► dossier de faits ─► rapport FR (template | LLM sous garde)
    ─► assistant (intentions fréquentes en template, sinon LLM sous garde)
```

Le même moteur C produit les features d'entraînement (sur tout le flux) et celles de l'inférence (micro-lots). Il n'y a donc aucun écart entre les deux.

## Ce que montrent les données (HI-Small)
Les chiffres de cette section ont été vérifiés pendant l'exploration ; les scripts de préparation les recalculent dans `data/processed/HI-Small/meta.json`.

- L'ACH porte 86,6 % des opérations illicites. Reinvestment, Wire et les opérations multi-devises n'en contiennent aucune.
- Les taux de change du simulateur sont fixes. Ils sont dérivés des opérations multi-devises du train (résidu médian 4·10⁻⁷), sans source externe.
- **Les montants se conservent le long des chaînes de blanchiment.** Sur les relais CYCLE, RANDOM, STACK et SCATTER-GATHER, le ratio montant réémis / montant reçu (en USD) a une médiane de 1,000.
- Les tentatives durent de 24 à 150 h (médiane selon la typologie). D'où des fenêtres de 6 h, 24 h et 96 h.
- 38 % des opérations illicites sont hors motifs. La traîne du 11 au 18/09 ne contient que des fins de motifs, avec environ 60 % d'illicites. Elle est donc évaluée à part.

## Moteur de features (C99, `aml/engine/aml_engine.c`)
Chaque transaction est traitée en deux temps : ses features sont calculées à partir de l'état, qui ne contient que les transactions passées, puis l'état est mis à jour avec elle. La causalité tient à la construction du moteur, et des tests la vérifient.

| Famille | Contenu | Typologies visées |
|---|---|---|
| `tx_` intrinsèques | montant USD, format, devises, change, auto-virement, même banque, heure, montant rond | – |
| `kyc_` connaissance client | type d'entité, nombre de comptes de l'entité, pays de la banque, transfrontalier | – |
| `s_` / `r_` profils | âge du compte, délais, moyennes glissantes 6 h/24 h/96 h (nombres, montants), z-score (Welford), part d'ACH, contreparties distinctes | fan-in, fan-out |
| `p_` relation | historique de la paire, arête inverse | cycle de 2 |
| `g_` degrés fenêtrés | fan-out et fan-in sur 24 h et 96 h (graphe ACH) | fan, gather-scatter |
| `f_` **conservation des flux** | appariement du montant avec les entrées récentes de l'émetteur, profondeur et âge de la chaîne de passage (propagée d'arête en arête) | stack, random, cycle, scatter-gather |
| `m_` motifs | cycles temporels refermés par l'arête (DFS bornée, montants à ±30 %), scatter-gather, chemins à deux sauts, recouvrement biparti | cycle, scatter-gather, bipartite |
| `z_` propagation (étape 2) | scores des transactions passées des deux comptes | toutes |
| `h_` revue J+1 | ce qui s'est passé après l'opération, avant la décision de minuit (réémission en aval, cycle complété…) | toutes |

**Mémoire bornée** :
- 136 octets d'état par compte ;
- table des paires avec éviction ;
- anneau d'arêtes de taille fixe ;
- parcours d'adjacence plafonnés et DFS à budget.

Les comptes « hubs » ne peuvent donc pas faire exploser le coût d'une transaction.

**Portabilité** :
- compilation sans `-march` ni intrinsèques, avec `-ffp-contract=off` ;
- version aarch64 compilée en croisé par `scripts/build_engine.sh aarch64`, via zig fourni par pip.

## Évaluation
- **Découpage temporel** : train du 01 au 06/09, validation du 07 au 08/09, test du 09 au 10/09, traîne du 11 au 18/09 évaluée à part.
  Le « protocole littérature » (test du 09/09 à la fin, traîne comprise) reproduit le découpage par jours de Multi-GNN et n'est utilisé que pour se comparer à l'état de l'art.
- **Choix sur la validation uniquement** : seuil, hyperparamètres et ablations. Le test n'est lu qu'une fois par configuration figée, et chaque lecture est consignée dans `results/eval_*`.
- **Tests anti-fuite** (`tests/test_engine.py`) :
  - les features de la transaction i sont identiques, que le flux s'arrête en i ou continue ;
  - perturber le futur ne change pas le passé ;
  - la propagation est causale ;
  - l'API du moteur ne reçoit aucune étiquette ;
  - le moteur C est identique à un oracle Python sur 180 000 transactions.
- **Métriques** : précision, rappel, F1 au seuil de validation (IC 95 % par bootstrap), PR-AUC, précision@k par jour, tentatives détectées et délai par typologie, rappel hors motifs.

## Techniques empruntées à d'autres domaines
| Technique | Domaine d'origine | Usage ici |
|---|---|---|
| Statistiques amorties (EWMA multi-échelles, Welford) | détection d'intrusion réseau, analyse numérique | profils en O(1), mémoire constante |
| Énumération de motifs temporels en flux | fouille de graphes (2SCENT, GFP) | cycles, scatter-gather |
| Loi des nœuds (Kirchhoff), appariement de montants | physique | comptes de passage |
| Suivi de provenance (*taint analysis*) | forensique blockchain, lignage de données | profondeur et âge de chaîne |
| Lissage à retard fixe | traitement du signal | revue J+1 |
| Classification collective, stacking temporel | apprentissage relationnel | propagation des scores (étape 2) |
| Regroupement de détections en pistes | vision, suivi multi-objets | alertes → cas |
| Compilation de modèle (arbres → tableaux C) | systèmes embarqués | inférence en µs, sans dépendance |
| Génération à emplacements (*slot filling*) | génération de langue, dialogue | LLM qui n'écrit aucun chiffre |
| Choix de cœur big.LITTLE | informatique mobile | énergie par transaction sur A55 et sur A76 |

## Résultats
Chiffres tirés de `results/*.json` (`python -m aml summarize` reproduit les tableaux). Machine : Xeon W-1270, un modèle livré (seed 42).

### Détection (test, modèle strict figé, évalué une seule fois)
| Période | Précision | Rappel | F1 [IC95] | PR-AUC | Alertes |
|---|---|---|---|---|---|
| Test 09–10/09 (traîne exclue) | 0,855 | 0,528 | **0,653** [0,624–0,681] | 0,638 | 591 |
| Traîne 11–18/09 (évaluée à part) | 0,969 | 0,861 | 0,912 [0,896–0,927] | 0,984 | 582 |
| Protocole littérature (09/09 → fin) | 0,911 | 0,664 | **0,768** [0,750–0,786] | 0,787 | 1 173 |

Source : `results/eval_strict_test-tail-test_lit_20261027-065943.json`. Seuil fixé au F1 maximal de la validation.
Sur la validation, 5 seeds donnent un F1 de 0,669 ± 0,006 et une PR-AUC de 0,650 ± 0,002 (`results/val_model_strict.json`).

### Variantes (même protocole, test lu une seule fois par configuration)
| Modèle | Test 09–10/09 F1 [IC95] | Traîne F1 | Protocole littérature F1 [IC95] | Validation F1 |
|---|---|---|---|---|
| strict (livré) | 0,653 [0,624–0,681] | 0,912 | 0,768 [0,750–0,786] | 0,680 |
| strict + cascade (tel qu'en ligne) | 0,648 [0,619–0,675] | 0,906 | 0,763 [0,746–0,781] | 0,674 |
| **étape 2 : propagation des scores** (causale) | **0,662** [0,635–0,689] | 0,939 | **0,788** [0,771–0,804] | 0,683 |
| revue J+1 (données jusqu'à minuit) | 0,653 [0,624–0,679] | 0,926 | 0,775 [0,757–0,792] | 0,686 |

Sources : `results/eval_*_test-tail-test_lit_*.json`.
- L'étape 2 (classification collective : propagation des scores passés des comptes) est la meilleure configuration : +0,009 de F1 sur le test, +0,020 sur le protocole littérature, sans aucune information future. Elle est aussi la plus stable : F1 de validation 0,683 ± 0,001 sur 3 seeds, contre 0,669 ± 0,006 sur 5 seeds pour le modèle strict.
- La revue J+1 n'apporte rien sur le test propre. Elle aide un peu sur la traîne, où les tentatives se complètent après coup.
- **Modèle temps réel livré : strict + cascade.** L'étape 2 a appris ses scores propagés sans cascade. Elle doit donc tourner sans cascade, avec deux forêts complètes par transaction. Son débit n'a pas été mesuré ; une seule forêt complète sans cascade fait 6 366 tx/s sur x86, et le rythme d'arrivée du jeu est d'environ 6 tx/s. C'est la configuration de précision, plus coûteuse.

### Comparaison à l'état de l'art (même découpage par jours, traîne comprise)
| Méthode | HI-Small F1 (%) | LI-Small F1 (%) | Information future utilisée |
|---|---|---|---|
| LightGBM, features de base (GFP) | 21,3 | – | non |
| GFP + XGBoost, lots de 128 | 63,2 | 27,3 | contenu du lot |
| Multi-PNA | 68,2 | 33,1 | tout le graphe de test |
| FraudGT | 67,9 | 28,6 | tout le graphe de test |
| SALT-Trans | 71,6 | – | tout le graphe de test |
| MEGA-GenAgg | 74,9 | 46,3 | tout le graphe de test |
| **Ce travail, strict** | **76,8** | **41,8** | **aucune** |
| **Ce travail, étape 2** | **78,8** | – | **aucune** |

En flux strictement causal, l'étape 2 dépasse sur HI-Small tous les résultats publiés trouvés. Sur LI-Small, le modèle strict dépasse GFP, FraudGT et Multi-GNN, mais reste derrière MEGA-GNN, qui voit le graphe de test (`results/eval_li_strict_test-tail-test_lit_*.json` ; LI-Small : 3 seeds en validation, F1 0,329 ± 0,004).

Précautions de lecture :
- les articles donnent une moyenne sur 5 seeds, ici c'est un seul modèle avec un IC95 par bootstrap ;
- leur seuil est souvent 0,5, le nôtre est fixé sur la validation ;
- GFP utilise un découpage 60/20/20 en nombre de transactions.

### Usage par l'analyste (test 09–10/09)
- **Précision des k premières alertes de chaque jour** : 1,0 pour k = 10, 25, 50 et 100.
- **Cas** : 272 au total (165 puis 107 par jour). 71 % contiennent au moins une opération illicite. La précision des 25 premiers cas de chaque jour vaut 1,0.
- **Tentatives détectées** :

| Typologie | Détectées |
|---|---|
| FAN-OUT | 20/20 |
| SCATTER-GATHER | 22/22 |
| CYCLE | 21/22 |
| STACK | 21/22 |
| GATHER-SCATTER | 34/37 |
| BIPARTITE | 14/16 |
| RANDOM | 15/17 |
| FAN-IN | 14/18 |

- **Délai de détection** : médiane de 0 à 48 h depuis le début réel de la tentative, selon la typologie.
- **Rappel en transactions** : 86 % sur les motifs, 0,5 % hors motifs. C'est la limite structurelle d'une détection en flux strict.
- **Typologie nommée** (`results/eval_typology_test_*.json`) : correcte pour 61 % des cas qui contiennent une opération illicite (193 cas). Transaction par transaction, l'exactitude est de 0,54 sur les illicites détectées. Les preuves structurelles (circuit, intermédiaires, éventail) complètent cette typologie probable dans le rapport.
  Écart de procédure : un premier fichier de cette évaluation sur le test a été effacé, puis refait à l'identique avec en plus le niveau cas.

### Démonstration (journée du 09/09, `results/demo_2022-09-09.json`)
- État restauré depuis l'instantané en 0,1 s.
- 654 467 transactions rejouées en **5,3 s** (moteur à 141 000 tx/s, cascade active) : 288 alertes, regroupées en 158 cas.
- Les 10 premiers cas ont chacun un rapport Markdown, JSON et HTML, avec un schéma des flux (`out/demo/index.html`).
- **Revue J+1 à minuit** (0,56 s) : 1 451 candidats re-scorés, 33 alertes nouvelles par rapport au temps réel, dont 14 vraies (étiquettes consultées après coup, uniquement pour la mesure).

### Ablations (validation, une seed)
| Configuration | PR-AUC | F1 |
|---|---|---|
| B0 règles expertes (seuils réglés sur la validation) | — | 0,333 |
| B1 features de la transaction seules | 0,075 | 0,162 |
| + profils de compte et relation | 0,612 | 0,643 |
| + degrés fenêtrés | 0,607 | 0,635 |
| + motifs (cycles, scatter-gather) | 0,608 | 0,643 |
| + **conservation des flux** | 0,625 | 0,674 |
| + KYC (modèle complet) | 0,653 | 0,680 |
| complet sans conservation des flux | 0,643 | 0,660 |
| complet sans motifs | 0,656 | 0,679 |

La conservation des flux apporte environ 2 points de F1. Les motifs explicites n'apportent rien de plus en score, mais ils servent de preuves dans les rapports et de signaux pour la typologie.

### Cascade (validation)
La mini-forêt (60 arbres, 10,7 Ko) écarte **99,3 %** des transactions avant la forêt complète (1 145 arbres). Elle perd 7 alertes sur 621 : F1 de 0,680 à 0,674. La courbe complète est dans `results/val_cascade_strict.json`.

## Frugalité
Tout est mesuré par `python -m aml bench` (un thread, résultats dans `results/bench_*.json`) :
- débit et latence par transaction (p50, p99, p99,9, mesurées côté C) ;
- latence de bout en bout depuis Python ;
- RAM (état du moteur compté octet par octet, RSS) ;
- tailles (forêt, bibliothèque, état sur disque) ;
- énergie (RAPL si lisible, sinon puissance relevée au wattmètre : `--watts-idle`, `--watts-load`).

### Mesures x86 (Xeon W-1270, un thread, `results/bench_ORC_x86_64.json`)
| Mesure | Valeur |
|---|---|
| Débit, flux complet (5,08 M tx, scoring compris, cascade active) | **214 000 tx/s** (23,7 s) |
| Latence moteur par transaction : p50 / p99 / p99,9 | **3,6 µs** / 10,7 µs / 227 µs |
| Latence d'un appel Python (lot de 1) : p50 / p99 | 78 µs / 330 µs |
| Coût par transaction en lots de 1 024 (p50) | 5,0 µs |
| État du moteur (comptes, paires, anneau) | 166 Mio (170 Mo sur disque) |
| RSS du processus (dont pages de fichiers en mmap) | 399 Mio (182 Mio) ; mémoire anonyme 216 Mio |
| Forêt complète, compressée sur disque / mini-forêt de la cascade / bibliothèque C | 1,6 Mo / 10,7 Ko / 43 Ko |
| Transactions écartées par la cascade | 99,3 % |
| Énergie | non mesurée (RAPL illisible sans droits root) ; mesure au wattmètre sur l'OrangePi |

Le p99,9 (227 µs) correspond aux 0,7 % de transactions qui passent par la forêt complète (1 145 arbres).

**Effet de la cascade, même machine** (`results/bench_ORC_x86_64_sans_cascade.json`) :

| | Débit | Latence p50 | F1 test (`results/eval_strict*_test-*`) |
|---|---|---|---|
| sans cascade | 6 366 tx/s | 162 µs | 0,653 |
| avec cascade | 214 055 tx/s | 3,6 µs | 0,648 |

Le débit est multiplié par **33,6** pour −0,005 de F1.

**Taille de forêt contre qualité** (validation, seuil re-choisi à chaque taille, coût mesuré en C ; `results/val_pareto_strict.json`) :

| Arbres | 50 | 100 | 200 | 300 | 500 | 800 | 1 145 |
|---|---|---|---|---|---|---|---|
| F1 | 0,617 | 0,650 | 0,663 | 0,667 | 0,671 | 0,675 | 0,680 |
| PR-AUC | 0,610 | 0,633 | 0,638 | 0,640 | 0,645 | 0,650 | 0,653 |
| µs par transaction | 1,3 | 4,6 | 15,0 | 30,1 | 59,0 | 102,0 | 158,0 |

La cascade évite ce compromis : la forêt complète n'est évaluée que sur 0,7 % des transactions.

Leviers mesurés :
- **Cascade à la Viola-Jones** : une mini-forêt (quelques dizaines d'arbres) écarte les transactions manifestement licites. La forêt complète ne tourne que sur le reste. Le seuil, fixé sur la validation, conserve 99 % des alertes du modèle complet ; la courbe complète est dans `results/val_cascade_*.json`.
- **Instantané de l'état** (`aml_snapshot` / `aml_restore`) : la mise en route ne rejoue pas l'historique.
- **Graphe de motifs restreint à l'ACH** ; 136 octets d'état par compte ; forêt compacte (nœuds de 12 octets en ordre préfixe, seuils float32 exacts).

## Explicabilité
Chaque alerte est justifiée à trois niveaux, tous visibles dans le rapport.
1. **Raisons du modèle** :
   - contributions TreeSHAP (lightgbm `pred_contrib`), ou Saabas sans lightgbm ;
   - traduites en phrases françaises avec la valeur observée et son rang parmi les transactions d'apprentissage, par exemple « le destinataire a reçu de 6 émetteurs distincts en 96 h — valeur plus élevée que 99,1 % des transactions d'apprentissage » ;
   - le contexte (format, montant) est séparé du comportement et de la structure.
2. **Indicateurs à seuil fixe**, lisibles sans modèle : éventail sortant ou entrant, fonds de passage, chaîne de passage, circuit fermé, dispersion puis collecte.
3. **Preuves reconstituées** à partir des seules opérations antérieures : chemin du circuit, chaîne de passage (montants quasi identiques de relais en relais), bénéficiaires de l'éventail.

La **fidélité** des raisons est mesurée (`results/val_explain_strict.json`, 621 alertes de validation) :
- remplacer les 3 premières raisons par la médiane d'apprentissage fait repasser **99,8 %** des alertes sous le seuil (baisse médiane du score : 0,98) ;
- les indicateurs à seuil fixe couvrent 57 % des vraies alertes et 14 % des fausses : une alerte sans indicateur mérite plus de prudence.

La première contribution brute est presque toujours le contexte (virement ACH). Le rapport présente donc d'abord les raisons de comportement et de structure, puis le contexte sur une ligne à part.

## Rapport de type TRACFIN et LLM
- Un dossier de faits JSON par cas alimente un template Jinja2 (`aml/templates/rapport.md.j2`) qui suit les rubriques publiées par Tracfin : déclarant, personnes et comptes, opérations, analyse du soupçon, éléments d'appréciation, limites. Sources dans `data/SOURCES.md`.
- **Garde du LLM** (`aml/llm.py`) :
  - le modèle ne voit jamais une valeur : nombres, dates, montants et identifiants sont remplacés par des jetons `⟦Vn⟧` ;
  - la réponse est rejetée si elle contient un chiffre, un jeton inconnu ou trop peu de jetons repris ;
  - le code réinjecte ensuite les valeurs exactes ;
  - en cas de rejet ou sans LLM, le template est utilisé tel quel.
- **Assistant** (`python -m aml ask`) : les questions fréquentes (raisons, comptes, montants, chronologie, typologie, suite à donner, fiabilité) reçoivent une réponse par template, instantanée. Les autres passent par le LLM sous la même garde.

### Banc LLM (x86, 4 threads, 10 dossiers réels de la démo)
| Modèle (Q4_K_M) | Jetons non typés : acceptation | Jetons typés : acceptation | Latence médiane |
|---|---|---|---|
| Qwen3.5-0.8B | 90 % (jetons mal placés) | **100 %** | 25 s |
| Qwen3.5-2B | 100 % | 90 % | 47 s |
| LFM2.5-1.2B | 0 % (ne reprend aucune valeur) | 0 % | 23 s |

Sources : `results/llm_bench_ORC_x86_64_jetons_non_types.json` et `results/llm_bench_ORC_x86_64.json`.
- Typer les jetons (`⟦DATE_n⟧`, `⟦COMPTE_n⟧`, `⟦MONTANT_n⟧`…) a supprimé les contresens du petit modèle, qui mettait par exemple une date à la place d'un compte.
- Qwen3.5-0.8B devient le modèle par défaut.
- Les textes acceptés restent très proches du template : le LLM apporte peu au rapport, alors qu'un rapport par template se produit en quelques millisecondes. Le template reste donc le mode par défaut, et le LLM une option (`--llm`), surtout utile à l'assistant pour les questions libres.

## Limites
- Données synthétiques (AMLworld) : les typologies et la conservation des montants reflètent le générateur.
- Les opérations illicites hors motifs (38 %) sont isolées et difficiles à détecter en flux strict.
- Seuil fixé au F1 maximal de validation ; en production, il se règle sur la capacité des analystes (précision@k).
