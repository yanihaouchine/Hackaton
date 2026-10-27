# Solution par réseau de neurones (Anas, Youssef)

## 1. À quoi sert ce dossier

Comparer honnêtement un **réseau de neurones (MLP)** à la **forêt d'arbres de Yani** (`arbre/`).
Pour que la comparaison soit juste, le MLP utilise **exactement** :

- **les mêmes features** : les 96 colonnes du modèle « strict » de Yani, calculées par son moteur C (empreinte `03a26672a4ef`) ;
- **le même découpage temporel** : train du 01 au 06/09, validation les 07 et 08/09, test les 09 et 10/09, traîne du 11 au 18/09 à part ;
- **le même tirage** : toutes les transactions illicites du train et 25 % des licites (seed 42).

On ne copie pas ce code : `nn/data.py` **importe celui de `arbre/`**, en lecture seule. Si Yani corrige quelque chose, on en profite automatiquement.

Ce dossier accueille aussi deux **livrables attendus par CGI**, indépendants du modèle de détection :
- `rapport/` : le rapport inspiré de la déclaration de soupçon TRACFIN, généré à partir d'un dossier d'alerte (JSON) ;
- `interface/` : l'interface de suivi des alertes pour la démo (Streamlit, hors du chemin de détection).

Ils lisent seulement des **alertes et leurs raisons**, au format des dossiers de `arbre/` (`Reporter.facts`). Forêt, MLP ou combinaison : tout modèle qui produit ce format convient.

## État des 6 livrables du kick-off (27/10, 15 h 45)

| Livrable | Ce qui existe | Ce qui manque |
|---|---|---|
| Graphe des transactions visualisé | `arbre/aml/casegraph.py` : schéma SVG statique d'un cas (démo de Yani) ; **`neurones/interface/`** : graphe des comptes de chaque cas, compte principal en évidence | — |
| Modèle de détection | `arbre/` : forêt strict + cascade, F1 test 0,653 ; `neurones/` : MLP, F1 test 0,606 (Youssef l'améliore) ; `anas/pipeline` : règles seules (archive) | le choix final du modèle (équipe) |
| Alertes en temps réel | `arbre/aml/stream.py` (`Monitor.replay`) et `python -m aml demo` (tableau en direct dans le terminal) ; `rapport/make_alerts.py` rejoue une journée sans écrire dans `arbre/` | un affichage en direct dans l'interface (facultatif) |
| Rapports générés automatiquement | `arbre/aml/report.py` (6 sections, LLM optionnel sous garde) ; **`neurones/rapport/`** (8 sections, sans LLM, garde-fou numérique, testé) | la version avec LLM de `neurones/rapport/` (étape 4) |
| Interface de suivi des alertes | `arbre/` : pages HTML statiques ; **`neurones/interface/`** : file triée par score, raisons, graphe, rapport, décision « Valider » ou « Classer sans suite » | — |
| Mesures de frugalité | `arbre/results/bench_*.json` (débit, latences, RAM, tailles) ; `neurones/results/bench_mlp_x86_64.json`, `rapport_bench.json` | l'énergie (wattmètre sur l'OrangePi, RAPL illisible ici) |

## 2. Ce qui est fait

| Fichier | Rôle |
|---|---|
| `nn/data.py` | charge les features, le découpage et le tirage de `arbre/`, puis normalise : one-hot pour les 8 codes (pays, devise, heure…), `signe·log(1+|x|)` puis centrage-réduction pour les autres. Les statistiques sont calculées **sur le train uniquement**. 96 features deviennent 220 entrées. |
| `nn/model.py` | le MLP **220 → 128 → 64 → 1** (ReLU, dropout 0,1, 36 609 paramètres). La normalisation est **intégrée au modèle** : le modèle livré prend directement les features brutes. |
| `scripts/train.py` | entraînement (Adam, lots de 4 096). Le poids des fraudes est choisi sur la validation (1, 10 ou 100). **Arrêt précoce sur la loss de validation** (patience 3). Seuil = F1 maximal sur la validation. 3 seeds. |
| `scripts/export.py` | export **ONNX** (torch reste hors du livrable) et vérification que ONNX donne les mêmes scores que torch. |
| `nn/explain.py` | **raisons d'une alerte** par « gradient × entrée », au **même format que les raisons de Yani** : ses phrases en français et ses rangs sont réutilisés. **Déjà fait.** |
| `scripts/explain_eval.py` | fidélité des raisons, même protocole que `arbre/`. |
| `scripts/confusion.py` | matrices de confusion du MLP **et** de la forêt, chacun à son seuil de validation. La forêt est rejouée sur notre machine et recoupée avec les chiffres publiés par Yani. |
| `scripts/bench.py` | taille, latence et débit du modèle seul : MLP contre forêt complète, et forêt avec cascade (configuration livrée par Yani). |
| `scripts/evaluate.py` | évaluation **finale** sur le test, la traîne et le protocole littérature. Refuse de tourner deux fois. |
| `scripts/plots.py` | images pour les slides, dans `results/figures/`. |
| `tests/` | 27 tests : normalisation, absence de fuite, normalisation torch identique à numpy, explications, rapport, interface. |
| `rapport/make_alerts.py` | rejoue une journée en flux avec la détection de `arbre/` (lecture seule, sans étiquette) et écrit les dossiers des cas les plus prioritaires dans `data/neurones/alertes/<date>/`. |
| `rapport/render.py`, `rapport/templates/rapport_tracfin.md.j2` | rapport **sans LLM** en 8 sections : identification du compte, résumé, typologie, motifs du soupçon, faits chiffrés, chronologie, comptes liés, recommandation. Le template **n'écrit aucun nombre** : il recopie les textes déjà formatés du dossier. |
| `rapport/guard.py` | garde-fou : un rapport qui contient un nombre absent du dossier est **refusé**. |
| `rapport/report.py` | génère le rapport d'un cas, ou de tous avec la mesure du temps. |
| `interface/app.py`, `interface/logic.py` | interface Streamlit : file des cas triée par score (filtres par typologie et statut), raisons de l'alerte, graphe des comptes (une flèche par opération), rapport TRACFIN téléchargeable, boutons « Valider » et « Classer sans suite ». Les décisions sont ajoutées à `data/neurones/decisions.jsonl` (rien n'est effacé). L'interface ne lit que les dossiers JSON : elle n'importe ni la détection ni le MLP. |
| `interface/run.sh` | lance l'interface en une commande (et produit les alertes si elles manquent). |

## 3. Résultats mesurés

Tous mesurés le 27/10/2026 sur un portable Intel Core i3-1115G4 (4 cœurs), seed 42. Chaque chiffre vient du fichier indiqué.

### Détection : MLP contre forêt de Yani

| Période | Modèle | Précision | Rappel | F1 [IC 95 %] | PR-AUC | Alertes |
|---|---|---|---|---|---|---|
| Validation (07–08/09) | MLP | 0,831 | 0,505 | 0,628 | 0,603 | 629 |
| | Forêt | 0,907 | 0,543 | 0,680 | 0,653 | 621 |
| **Test (09–10/09)** | **MLP** | 0,758 | 0,505 | **0,606** [0,578–0,632] | 0,586 | 637 |
| | **Forêt** | 0,854 | 0,528 | **0,653** [0,624–0,681] | 0,638 | 591 |
| Traîne (11–18/09) | MLP | 0,964 | 0,901 | 0,931 | 0,980 | 612 |
| | Forêt | 0,969 | 0,861 | 0,912 | 0,984 | 582 |

Sources : `results/confusion.json` (validation et test), `results/eval_mlp_test.json` (test, traîne, protocole littérature : le test n'a été lu **qu'une fois**). Les chiffres de la forêt proviennent de `arbre/results/eval_strict_test-tail-test_lit_20261027-065943.json`, et nous les retrouvons à l'unité près en rejouant la forêt (`forest_check_*` dans `results/confusion.json`).

**Conclusion : la forêt reste meilleure.** Sur le test, le MLP fait **0,606 de F1 contre 0,653** : les intervalles se touchent à peine, et la forêt a nettement moins de fausses alertes (86 contre 154). Le MLP ne fait mieux que sur la traîne, une période très particulière (59 % de fraudes).
Stabilité du MLP sur la validation, 3 seeds : F1 0,605 ± 0,017, PR-AUC 0,584 ± 0,014 (`results/train_mlp.json`).

### Matrices de confusion du test (seuil fixé sur la validation)

| | VP | FP | FN | VN | Accuracy |
|---|---|---|---|---|---|
| MLP | 483 | 154 | 473 | 861 682 | 99,93 % |
| Forêt | 505 | 86 | 451 | 861 750 | 99,94 % |
| Modèle « toujours licite » | 0 | 0 | 956 | 861 836 | **99,89 %** |

L'**accuracy est trompeuse** : un modèle qui ne signale jamais rien a 99,89 %, presque autant que nos modèles, tout en ratant 100 % des fraudes. On compare donc à la **précision, au rappel, au F1 et à la PR-AUC** (`results/confusion.json`).

### Frugalité : coût du modèle seul (le moteur de features C, commun aux deux, s'y ajoute)

| Modèle | Fichier | Latence p50 / p99 (1 transaction) | Débit (lots) |
|---|---|---|---|
| MLP (ONNX, 1 thread) | **156 Ko** | 25,6 µs / 49,7 µs | 458 903 tx/s |
| Forêt complète (1 145 arbres, C) | 1,6 Mo | 748 µs / 1 085 µs | 17 192 tx/s |
| **Forêt avec cascade** (configuration livrée par Yani) | 1,6 Mo | **10,7 µs / 34,6 µs** | **950 048 tx/s** |

Source : `results/bench_mlp_x86_64.json`. **Le MLP est 10 fois plus petit**, mais la forêt livrée (avec cascade) est **plus rapide** : seules 0,7 % des transactions passent par la grande forêt.
Énergie : **non mesurée** (RAPL illisible sans droits root sur ce portable).
Modèle ONNX identique à torch : écart maximal 1,4·10⁻⁷ (`results/export_mlp.json`).

### Explicabilité (validation, 629 alertes du MLP)

En remplaçant les k premières raisons d'une alerte par la valeur médiane du train, la part d'alertes qui repasse sous le seuil est :

| k | Raisons du MLP | k features au hasard (témoin) |
|---|---|---|
| 1 | 67,7 % | 4,8 % |
| 3 | **95,2 %** | 10,5 % |
| 5 | 99,5 % | 18,1 % |

Les raisons affichées portent donc bien la décision. Pour comparaison, la forêt (TreeSHAP, k = 3) atteint 99,8 % (`arbre/results/val_explain_strict.json`). Source : `results/explain_mlp.json`.

### Images pour les slides (`results/figures/`)

- `mlp_courbes.png` : loss (train et validation), accuracy (train et validation) et PR-AUC de validation, par epoch ;
- `confusion_val.png`, `confusion_test.png` : MLP et forêt côte à côte.

### Rapport TRACFIN (sans LLM)

Journée du 09/09/2022 rejouée en flux avec la détection de `arbre/` : 654 467 transactions, 288 alertes, 158 cas, ce qui est identique à la démo publiée par Yani (`arbre/results/demo_2022-09-09.json`). Les 30 cas les plus prioritaires ont chacun leur rapport.
Temps de génération d'un rapport : **0,51 ms** (p50), 5,9 ms (p99), sur 30 rapports ; garde-fou accepté pour les 30 (`results/rapport_bench.json`).

### Ressources de l'entraînement

Entraînement complet (3 poids × seed 42, puis 2 seeds) : 5 min 20 s, RAM maximale 8,5 Go (préparation de l'entraînement, pas l'inférence).

## 4. Qui fait quoi, et ce qui reste

Nouvelle répartition (27/10 après-midi) : Youssef et Anas ont chacun entraîné un MLP. **Youssef garde la partie modèle**,
Anas passe aux **livrables attendus par CGI**, qui fonctionnent quel que soit le modèle retenu.

| Qui | Fait | Reste à faire |
|---|---|---|
| Youssef | son MLP | **toute la partie modèle** (`nn/`, `scripts/train.py`…) : l'améliorer, le comparer à celui d'Anas et à la forêt, avec le même protocole. `nn/explain.py` existe déjà : inutile de le refaire. |
| Anas | premier MLP et ses mesures ; `rapport/` (rapport TRACFIN sans LLM, garde-fou, tests) ; `interface/` (suivi des alertes, graphe, rapport, décision) | le LLM qui reformule le rapport sous garde (si le temps le permet) ; slides des livrables et scénario de démo |
| Yani | `arbre/` : détection, rapports, démo, mesures | relire la PR ; `arbre/` reste en lecture seule pour nous |
| Ensemble | | choix du modèle final ; Pull Request `neurones → main`, relue par Yani |

## 5. Tout relancer, dans l'ordre

```bash
# Features de Yani (depuis la racine du dépôt, une fois par machine)
cd arbre
bash scripts/build_engine.sh            # compile le moteur C
../.venv/bin/python -m aml prepare      # CSV → flux trié (≈ 10 s)
../.venv/bin/python -m aml features     # 104 features causales → data/features/ (≈ 10 s)

# MLP (depuis neurones/)
cd ../neurones
../.venv/bin/python -m pytest                     # 13 tests
../.venv/bin/python scripts/check_data.py         # → results/data_summary.json
../.venv/bin/python scripts/train.py              # → artifacts/mlp.pt, results/train_mlp.json (≈ 5 min)
../.venv/bin/python scripts/export.py             # → artifacts/mlp.onnx, results/export_mlp.json
../.venv/bin/python scripts/confusion.py          # validation → results/confusion.json
../.venv/bin/python scripts/bench.py              # → results/bench_mlp_<machine>.json
../.venv/bin/python scripts/explain_eval.py       # → results/explain_mlp.json
../.venv/bin/python scripts/evaluate.py           # TEST : UNE SEULE FOIS → results/eval_mlp_test.json
../.venv/bin/python scripts/confusion.py --test   # matrices du test → results/confusion.json
../.venv/bin/python scripts/plots.py              # → results/figures/*.png

# Rapport TRACFIN (depuis neurones/), indépendant du modèle
../.venv/bin/python -m rapport.make_alerts                # rejoue le 09/09 → data/neurones/alertes/2022-09-09/ (≈ 11 s)
../.venv/bin/python -m rapport.report --case C08-0001     # → data/neurones/rapports/2022-09-09/C08-0001.md
../.venv/bin/python -m rapport.report --all               # les 30 rapports + results/rapport_bench.json

# Interface de démo (depuis la racine du dépôt) → http://localhost:8501
bash neurones/interface/run.sh                            # PORT=8502 bash … pour changer de port
```

Dépendances (dans le `.venv` de la racine) : numpy, polars, torch (CPU, entraînement uniquement), onnx, onnxruntime, matplotlib, jinja2 (rapport), streamlit 1.50.0 (démo uniquement : `pip install streamlit==1.50.0`).
Les modèles (`artifacts/`) ne sont pas versionnés : ils se régénèrent avec `train.py`, puis `export.py`.

**Règle d'or : le test ne s'évalue qu'une seule fois**, quand tout est figé. Le seuil, le poids des fraudes et l'arrêt précoce se choisissent **sur la validation uniquement**. `evaluate.py` refuse de réécrire `results/eval_mlp_test.json`. Relancer le test après avoir modifié le modèle, c'est tricher.

## 6. Règles git de l'équipe

- On travaille sur la branche **`neurones`** : `git switch neurones`.
- **Avant de commencer** : `git pull`.
- On ajoute **uniquement** notre dossier : `git add neurones/`. **Jamais `git add .`**, qui embarquerait les PDF (dont un contient un mot de passe) et des fichiers personnels.
- Petits commits, push fréquents.
- **Jamais de push direct sur `main`** : on passe par une **Pull Request `neurones → main`**, relue par l'équipe.
- On ne modifie pas `arbre/` sans prévenir Yani.
