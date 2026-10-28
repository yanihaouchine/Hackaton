# Solution par réseau de neurones (Anas, Youssef)

## 1. À quoi sert ce dossier

Comparer honnêtement un **réseau de neurones (MLP)** à la **forêt d'arbres de Yani** (`arbre/`).
Pour que la comparaison soit juste, le MLP utilise **exactement** :

- **les mêmes features** : les 96 colonnes du modèle « strict » de Yani, calculées par son moteur C (empreinte `03a26672a4ef`) ;
- **le même découpage temporel** : train du 01 au 06/09, validation les 07 et 08/09, test les 09 et 10/09, traîne du 11 au 18/09 à part ;
- **la même fonction de tirage et les mêmes métriques** : `mlp/common.py` **importe le code de `arbre/`**, en lecture seule.

Seul le modèle change. La version livrée est la **v2** (MLP 64→32). La v1 (MLP 128→64, Anas) reste documentée dans `results/v1/` ; son code est au commit `0c12204`.

Ce dossier accueille aussi deux **livrables attendus par CGI**, indépendants du modèle de détection :
- `rapport/` : le rapport inspiré de la déclaration de soupçon TRACFIN, généré à partir d'un dossier d'alerte (JSON) ;
- `interface/` : l'interface de suivi des alertes pour la démo (Streamlit, hors du chemin de détection).

Ils lisent seulement des **alertes et leurs raisons**, au format des dossiers de `arbre/` (`Reporter.facts`). Forêt, MLP ou combinaison : tout modèle qui produit ce format convient.

## État des 6 livrables du kick-off (27/10, 15 h 45)

| Livrable | Ce qui existe | Ce qui manque |
|---|---|---|
| Graphe des transactions visualisé | `arbre/aml/casegraph.py` : schéma SVG statique d'un cas (démo de Yani) ; **`neurones/interface/`** : graphe des comptes de chaque cas, compte principal en évidence | — |
| Modèle de détection | `arbre/` : forêt strict + cascade, F1 test 0,653 ; `neurones/` : MLP v2, F1 test 0,620 (v1 : 0,606) ; `anas/pipeline` : règles seules (archive) | le choix final du modèle (équipe) |
| Alertes en temps réel | `arbre/aml/stream.py` (`Monitor.replay`) et `python -m aml demo` (tableau en direct dans le terminal) ; `rapport/make_alerts.py` rejoue une journée sans écrire dans `arbre/` | un affichage en direct dans l'interface (facultatif) |
| Rapports générés automatiquement | `arbre/aml/report.py` (6 sections, LLM optionnel sous garde) ; **`neurones/rapport/`** (8 sections, garde-fou numérique, résumé reformulable par un LLM local sous garde, testé) | — |
| Interface de suivi des alertes | `arbre/` : pages HTML statiques ; **`neurones/interface/`** : file triée par score, raisons, graphe, rapport, décision « Valider » ou « Classer sans suite » | — |
| Mesures de frugalité | `arbre/results/bench_*.json` (débit, latences, RAM, tailles) ; `neurones/results/bench_arm64.json` (v1 : `results/v1/bench_mlp_x86_64.json`), `rapport_bench.json` | l'énergie sur l'OrangePi (wattmètre) ; mesurée par RAPL sur le portable (`results/energie_*`) |

## 2. Ce qui est fait

| Fichier | Rôle |
|---|---|
| `mlp/common.py` | chemins, lecture des features épinglées, accès en lecture seule au code de `arbre/`, écriture des résultats avec la provenance (date, commit, seed, machine). |
| `mlp/prep.py` | prétraitement **intégré au modèle** (le modèle livré prend les features brutes) : one-hot des catégories, heure en sinus/cosinus, indicateurs « absent » (-1) et « aucun appariement » (9), log1p des compteurs et délais, standardisation et écrêtage à ±5. Statistiques **du train uniquement**. 96 features → 210 entrées. |
| `mlp/train.py` | un entraînement : AdamW, lots de 4 096, arrêt précoce sur la **PR-AUC de validation** (patience 5), seuil au F1 maximal de validation, export **ONNX** vérifié contre torch, poids torch pour les explications. |
| `mlp/sweep.py` | budget d'hyperparamètres **fixé d'avance** : 3 tailles × 2 taux de licites sur la seed 42, puis 5 seeds pour la meilleure. |
| `mlp/tiebreak.py` | départage 64→32 / 256→128 selon une règle écrite avant d'en voir le résultat (voir `CLAUDE.md`). |
| `mlp/evaluate.py` | évaluation **finale** sur le test, la traîne et le protocole littérature, via l'ONNX livré, même protocole que `arbre/`. Chaque lecture écrit un fichier horodaté. |
| `mlp/summarize_test.py` | moyenne ± écart-type des 5 seeds sur le test, à partir des fichiers d'évaluation (ne relit pas le test). |
| `mlp/explain.py`, `mlp/explain_eval.py` | **raisons d'une alerte** par « gradient × entrée » (méthode de la v1), au format des raisons de Yani ; fidélité mesurée par son protocole. |
| `mlp/confusion.py` | matrices de confusion du MLP **et** de la forêt, rejouée sur notre machine et recoupée avec les chiffres publiés par Yani. |
| `mlp/bench.py` | taille, latence et débit du modèle seul : MLP contre forêt complète et forêt avec cascade (protocole de la v1). |
| `mlp/plots.py` | images pour les slides, dans `results/figures/`. |
| `scripts/build_db.py` | base DuckDB d'exploration (`data/neurones/aml.duckdb`), jamais utilisée en inférence. |
| `tests/` | 39 tests : prétraitement (sentinelles, one-hot, statistiques du train seul, écrêtage, heure cyclique), explications, rapport, interface, garde du LLM (avec un faux serveur compatible Ollama). |
| `rapport/make_alerts.py` | rejoue une journée en flux avec la détection de `arbre/` (lecture seule, sans étiquette) et écrit les dossiers des cas les plus prioritaires dans `data/neurones/alertes/<date>/`. |
| `rapport/render.py`, `rapport/templates/rapport_tracfin.md.j2` | rapport **sans LLM** en 8 sections : identification du compte, résumé, typologie, motifs du soupçon, faits chiffrés, chronologie, comptes liés, recommandation. Le template **n'écrit aucun nombre** : il recopie les textes déjà formatés du dossier. |
| `rapport/guard.py` | garde-fou : un rapport qui contient un nombre absent du dossier est **refusé**. |
| `rapport/report.py` | génère le rapport d'un cas, ou de tous avec la mesure du temps (`--llm` pour le LLM). |
| `rapport/llm.py` | **LLM local sous garde** (Ollama ou llama-server, API compatible OpenAI, bibliothèque standard seulement). Le LLM ne reformule que le **résumé**. Il ne voit **aucun nombre** : chaque valeur devient un marqueur `⟦F1⟧`. Sa réponse est **rejetée** si elle contient un chiffre, un marqueur inconnu, ou si elle perd le montant total ; sinon le code réinjecte les valeurs et le garde-fou revérifie tout le rapport. Sans serveur ou en cas de rejet : résumé du template. |
| `interface/app.py`, `interface/logic.py` | interface Streamlit : file des cas triée par score (filtres par typologie et statut), raisons de l'alerte, graphe des comptes (une flèche par opération), rapport TRACFIN téléchargeable (option « Reformuler avec le LLM local »), boutons « Valider » et « Classer sans suite ». Les décisions sont ajoutées à `data/neurones/decisions.jsonl` (rien n'est effacé). L'interface ne lit que les dossiers JSON : elle n'importe ni la détection ni le MLP. |
| `interface/run.sh` | lance l'interface en une commande (et produit les alertes si elles manquent). |

## 3. Résultats mesurés

Mesurés le 27/10/2026 sur un Mac Apple M5 (sauf mention contraire), seed 42 pour le modèle livré. Chaque chiffre vient du fichier indiqué.

### Choix du modèle (validation uniquement)

| Configuration | Paramètres | PR-AUC val (5 seeds) | F1 val (5 seeds) |
|---|---|---|---|
| **v2 : 64 → 32**, 100 % des licites | **15 617** | **0,627 ± 0,002** | **0,653 ± 0,006** |
| v2 : 256 → 128, 100 % des licites | 87 041 | 0,629 ± 0,003 | 0,652 ± 0,006 |
| v1 : 128 → 64, 25 % des licites (3 seeds) | 36 609 | 0,584 ± 0,014 | 0,605 ± 0,017 |
| Forêt strict de Yani | — | 0,650 ± 0,002 | 0,669 ± 0,006 |

Sources : `results/val_mlp_tiebreak.json`, `results/val_mlp_sweep.json`, `results/v1/train_mlp.json`, `arbre/README.md`. Le grand réseau ne fait pas significativement mieux : on garde le petit (règle de départage dans `CLAUDE.md`).

### Détection : MLP contre forêt de Yani

| Période | Modèle | Précision | Rappel | F1 [IC 95 %] | PR-AUC | Alertes |
|---|---|---|---|---|---|---|
| **Test (09–10/09)** | **MLP v2** | 0,743 | 0,531 | **0,620** [0,589–0,646] | 0,606 | 684 |
| | MLP v1 | 0,758 | 0,505 | 0,606 [0,578–0,632] | 0,586 | 637 |
| | **Forêt** | 0,854 | 0,528 | **0,653** [0,624–0,681] | 0,638 | 591 |
| Traîne (11–18/09) | MLP v2 | 0,953 | 0,951 | 0,952 [0,940–0,964] | 0,980 | 654 |
| | Forêt | 0,969 | 0,861 | 0,912 [0,896–0,927] | 0,984 | 582 |
| Protocole littérature | MLP v2 | 0,845 | 0,702 | 0,767 [0,749–0,784] | 0,775 | 1 338 |
| | Forêt | 0,911 | 0,664 | 0,768 [0,750–0,786] | 0,787 | 1 173 |

Sur 5 seeds, le MLP v2 fait **F1 test 0,626 ± 0,011** (`results/test_mlp_seeds.json`). Le test n'a été lu **qu'une fois**, configuration figée (`results/eval_mlp_h64-32_n100_s42_test-tail-test_lit_20261027-153049.json`). Chiffres de la forêt : `arbre/results/eval_strict_test-tail-test_lit_20261027-065943.json`, retrouvés à l'unité près en rejouant la forêt (`forest_check_*` dans `results/confusion.json`).

**Tentatives détectées sur le test** (au moins une alerte) : MLP v2 **168/174**, forêt 161/174. Par typologie (MLP / forêt) : CYCLE 22/21, GATHER-SCATTER 36/34, BIPARTITE 16/14, FAN-IN 16/14, RANDOM 16/15, STACK 21/21, SCATTER-GATHER 22/22, FAN-OUT 19/20.

**Conclusion : la forêt reste le modèle de détection.** Le MLP v2 a le même rappel, mais deux fois plus de fausses alertes (176 contre 86) : F1 0,620 contre 0,653. Il repère un peu plus de tentatives, fait mieux sur la traîne et égale la forêt sur le protocole littérature. Il améliore nettement la v1 (+0,014 de F1 test, 2,3 fois plus petit).

### Matrices de confusion du test (seuil fixé sur la validation)

| | VP | FP | FN | VN | Accuracy |
|---|---|---|---|---|---|
| MLP v2 | 508 | 176 | 448 | 861 660 | 99,93 % |
| Forêt | 505 | 86 | 451 | 861 750 | 99,94 % |
| Modèle « toujours licite » | 0 | 0 | 956 | 861 836 | **99,89 %** |

L'**accuracy est trompeuse** : un modèle qui ne signale jamais rien atteint 99,89 % tout en ratant 100 % des fraudes. On compare donc la **précision, le rappel, le F1 et la PR-AUC** (`results/confusion.json`).

### Frugalité : coût du modèle seul (le moteur de features C, commun aux deux, s'y ajoute)

| Modèle | Fichier | Latence p50 / p99 (1 transaction) | Débit (lots) |
|---|---|---|---|
| **MLP v2** (ONNX, 1 thread) | **74 Ko** | 14,5 µs / 18,2 µs | **1 679 995 tx/s** |
| Forêt complète (C) | 1,6 Mo | 322 µs / 395 µs | 29 182 tx/s |
| Forêt avec cascade (configuration livrée par Yani) | 1,6 Mo | **3,7 µs / 5,1 µs** | 1 598 584 tx/s |

Source : `results/bench_arm64.json` (Apple M5). **Le MLP est 21 fois plus petit** que la forêt et a le même débit que la forêt avec cascade ; la cascade garde la meilleure latence par appel (le MLP paie surtout le coût fixe d'un appel à onnxruntime). Mesures de la v1 sur un portable Intel Core i3-1115G4 : `results/v1/bench_mlp_x86_64.json` (machines différentes : seuls les rapports entre modèles d'une même machine se comparent).
Énergie : **non mesurée** (powermetrics demande les droits administrateur sur macOS). OrangePi : **non mesuré**.
ONNX identique à torch : écart maximal 1,3·10⁻⁷ (`results/val_mlp_h64-32_n100_s42.json`, champ `export`).

### Explicabilité (validation, 652 alertes du MLP v2)

En remplaçant les k premières raisons d'une alerte par la valeur médiane du train, la part d'alertes qui repasse sous le seuil est :

| k | MLP v2 | MLP v1 | k features au hasard (témoin, v2) |
|---|---|---|---|
| 1 | 50,6 % | 67,7 % | 4,6 % |
| 3 | **86,5 %** | 95,2 % | 12,3 % |
| 5 | 96,6 % | 99,5 % | 15,8 % |

Les raisons affichées portent la décision, mais un peu moins nettement qu'avec la v1 ; la forêt (TreeSHAP, k = 3) atteint 99,8 % (`arbre/results/val_explain_strict.json`). Source : `results/explain_mlp.json`, `results/v1/explain_mlp.json`.

### Images pour les slides (`results/figures/`)

- `mlp_courbes.png` : perte d'entraînement et PR-AUC de validation par époque, avec l'époque retenue et la forêt en référence ;
- `confusion_val.png`, `confusion_test.png` : MLP et forêt côte à côte.

### Énergie (RAPL, portable Intel Core i3-1115G4, 27/10)

Mesurée par le compteur d'énergie du processeur (RAPL), sur un seul thread. La consommation **au repos** (2,9 à 3,5 W) est retranchée, pour ne garder que l'énergie due au calcul. Deux essais, d'où une fourchette (le portable varie d'environ 15 % d'un essai à l'autre).

| Travail mesuré | Durée | Énergie nette | Par unité |
|---|---|---|---|
| Détection en flux d'une journée (09/09 : 654 467 tx, 288 alertes ; forêt + cascade de `arbre/`) | 1,2 s | 11 à 13 J | **17 à 20 µJ par transaction** |
| Rapports sans LLM (6 000 rapports) | 3,3 s | 30 à 36 J | **4,9 à 5,9 mJ par rapport** |
| Rapports avec LLM | — | **non mesuré** (GPU de DeepEirb, hors de portée de RAPL) | — |

Ordre de grandeur : détecter toute une journée de transactions coûte autant qu'une ampoule LED de 10 W allumée **à peine plus d'une seconde**.
Sources : `results/energie_anas-HP-Laptop-15-dw3xxx_essai1.json` et `_essai2.json` ; script `mesures/energie.py` (capteur à débloquer : `sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj`).

### Rapport TRACFIN (sans LLM)

Journée du 09/09/2022 rejouée en flux avec la détection de `arbre/` : 654 467 transactions, 288 alertes, 158 cas, ce qui est identique à la démo publiée par Yani (`arbre/results/demo_2022-09-09.json`). Les 30 cas les plus prioritaires ont chacun leur rapport.
Temps de génération d'un rapport **sans LLM** : **0,57 ms** (p50), 6,2 ms (p99), sur 30 rapports ; garde-fou accepté pour les 30 (`results/rapport_bench.json`).
**Avec LLM** (résumé seulement), mesuré le 27/10 sur les 30 rapports, Ollama de **DeepEirb** (RTX 6000 Ada partagée) appelé depuis le portable par un tunnel SSH (le temps inclut le réseau) :

| Modèle (Ollama) | Taille | Temps par rapport (p50 / p99) | Résumés acceptés par la garde |
|---|---|---|---|
| **qwen3:1.7b** (choisi par défaut) | 1,4 Go | 1,23 s / 2,02 s | **27 / 30** |
| ministral-3:3b | 3,0 Go | 0,69 s / 0,82 s | 18 / 30 (12 marqueurs inventés) |
| gemma3:1b | 0,8 Go | 0,35 s / 0,63 s | 0 / 30 (marqueurs perdus, chiffres écrits) |
| qwen2.5:1.5b | 1,0 Go | 0,27 s / 0,72 s | 0 / 30 (chiffres écrits : il remplace ⟦TOTAL⟧ par un nombre) |

Sources : `results/rapport_bench_llm_<modèle>_<machine>.json`. Sans LLM : **0,57 ms**, soit **plus de 2 000 fois plus rapide**.
Ce que la garde a attrapé pendant la mise au point : un modèle qui écrivait « 5 comptes » à la place du marqueur numéro 5, un autre qui inventait « deux comptes » au lieu de cinq, un autre qui collait un pourcentage d'une raison à la typologie. D'où les **marqueurs nommés** (⟦TOTAL⟧ plutôt que ⟦C⟧) et le fait de ne confier au LLM que les raisons sans chiffre.
**Limite** : la garde vérifie que chaque nombre vient du dossier, pas que la phrase est juste. Les résumés acceptés relus reprennent les bonnes valeurs, mais reformulent peu. Conclusion pour le jury : **le template suffit**, le LLM est une option de confort, désactivée par défaut.

### Ressources de l'entraînement

Un entraînement 64→32 : environ 85 s (moyenne des 5 seeds, `results/val_mlp_tiebreak.json`). Balayage complet : RAM maximale 5,1 Go (préparation de l'entraînement, pas l'inférence).

## 4. Qui fait quoi, et ce qui reste

Répartition (27/10 après-midi) : Youssef et Anas ont chacun entraîné un MLP. **Youssef garde la partie modèle** (la v2 remplace la v1),
Anas passe aux **livrables attendus par CGI**, qui fonctionnent quel que soit le modèle retenu.

| Qui | Fait | Reste à faire |
|---|---|---|
| Youssef | MLP v2 : prétraitement, balayage d'hyperparamètres, départage, évaluation, intégration des outils de la v1 (`mlp/`) | mesurer sur l'OrangePi ou le Jetson ; brancher le MLP dans les rapports et la démo |
| Anas | MLP v1 et ses mesures ; `rapport/` (rapport TRACFIN, garde-fou, LLM sous garde, tests) ; `interface/` (suivi des alertes, graphe, rapport, décision) | slides des livrables et scénario de démo |
| Yani | `arbre/` : détection, rapports, démo, mesures | relire la PR ; `arbre/` reste en lecture seule pour nous |
| Ensemble | | choix du modèle final ; Pull Request `neurones → main`, relue par Yani |

## 5. Tout relancer, dans l'ordre

```bash
# Features de Yani (depuis la racine du dépôt, une fois par machine)
cd arbre
bash scripts/build_engine.sh            # compile le moteur C
../.venv/bin/python -m aml prepare      # CSV → flux trié
../.venv/bin/python -m aml features     # 104 features causales → data/features/

# MLP (depuis neurones/)
cd ../neurones
../.venv/bin/python -m pytest                                  # 39 tests (1 sauté sans les dossiers d'alerte)
../.venv/bin/python -m mlp.sweep                               # grille + 5 seeds → results/val_mlp_*.json (≈ 18 min)
for s in 43 44 45 46; do ../.venv/bin/python -m mlp.train --hidden 64,32 --neg-rate 1.0 --seed $s; done
../.venv/bin/python -m mlp.tiebreak                            # → results/val_mlp_tiebreak.json
../.venv/bin/python -m mlp.confusion                           # validation → results/confusion.json
../.venv/bin/python -m mlp.bench                               # → results/bench_<machine>.json
../.venv/bin/python -m mlp.explain_eval                        # → results/explain_mlp.json
../.venv/bin/python -m mlp.evaluate --models h64-32_n100_s4{2,3,4,5,6}   # TEST : UNE SEULE FOIS
../.venv/bin/python -m mlp.summarize_test                      # → results/test_mlp_seeds.json
../.venv/bin/python -m mlp.confusion --test                    # matrices du test → results/confusion.json
../.venv/bin/python -m mlp.plots                               # → results/figures/*.png
../.venv/bin/python scripts/build_db.py                        # optionnel : base DuckDB d'exploration

# Rapport TRACFIN (depuis neurones/), indépendant du modèle
../.venv/bin/python -m rapport.make_alerts                # rejoue le 09/09 → data/neurones/alertes/2022-09-09/ (≈ 11 s)
../.venv/bin/python -m rapport.report --case C08-0001     # → data/neurones/rapports/2022-09-09/C08-0001.md
../.venv/bin/python -m rapport.report --all               # les 30 rapports + results/rapport_bench.json
../.venv/bin/python -m mesures.energie                   # énergie RAPL → results/energie_<machine>.json (sudo chmod d'abord)
# Avec LLM, sur une machine de l'école (Ollama) : dans un autre terminal, `/opt/ollama/bin/ollama serve`, puis
../.venv/bin/python -m rapport.report --all --llm --llm-model qwen3:1.7b   # → results/rapport_bench_llm_<modèle>_<machine>.json
# Ou depuis le portable, LLM sur DeepEirb : ssh -N -J <login>@ssh.enseirb-matmeca.fr -L 11434:localhost:11434 <login>@deepeirb

# Interface de démo (depuis la racine du dépôt) → http://localhost:8501
bash neurones/interface/run.sh                            # PORT=8502 bash … pour changer de port
```

Dépendances (dans le `.venv` de la racine) : numpy, polars, torch (CPU, entraînement et explications), onnx, onnxruntime, matplotlib, duckdb (optionnel), jinja2 (rapport), streamlit 1.50.0 (démo uniquement : `pip install streamlit==1.50.0`).
Seuls les deux modèles livrés sont versionnés (`artifacts/h64-32_n100_s42/` et `artifacts/hist_h64-32_g16_s42/`, ≈ 340 Ko), pour que la démo tourne depuis un clone ; les autres (`artifacts/`) se régénèrent avec `mlp.train` et `mlp.train_hist` (déterministes : la seed 42 redonne un modèle identique).

**Règle d'or : le test ne s'évalue qu'une seule fois**, quand tout est figé. Taille, taux de licites, arrêt précoce et seuil se choisissent **sur la validation uniquement**. Relancer le test après avoir modifié le modèle, c'est tricher.

## 6. Règles git de l'équipe

- On travaille sur la branche **`neurones`** : `git switch neurones`.
- **Avant de commencer** : `git pull`.
- On ajoute **uniquement** notre dossier : `git add neurones/`. **Jamais `git add .`**, qui embarquerait les PDF (dont un contient un mot de passe) et des fichiers personnels.
- Petits commits, push fréquents.
- **Jamais de push direct sur `main`** : on passe par une **Pull Request `neurones → main`**, relue par l'équipe.
- On ne modifie pas `arbre/` sans prévenir Yani.
