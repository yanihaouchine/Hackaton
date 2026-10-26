# CLAUDE.md : hackathon AML basse consommation

Référence de toutes les sessions Claude Code. À tenir à jour (surtout « Décisions prises » et les TODO).

## Projet
Système de lutte anti-blanchiment (LCB-FT) **en flux** : rejouer IBM AML HI-Small comme un flux de transactions,
lever des alertes justifiées et produire pour chacune un **rapport inspiré de la déclaration de soupçon TRACFIN**.
Cible : machines de 4 à 7 W (la plus faible : OrangePi Zero 4, 4 GB, ARM A76/A55), mais aussi Jetson Orin, RTX 3060 et Xeon/A6000.

Grille de notation (rattacher chaque décision à une ligne) :
| Poids | Critère | Ce qu'on montre |
|---|---|---|
| 25 % | Pertinence métier | Alertes triées par priorité, typologie nommée, rapport exploitable par un analyste |
| 20 % | Qualité technique | F1, PR-AUC, précision@k sur un test temporel, baselines |
| 20 % | Frugalité | Latence p50/p99, débit, RAM max, taille sur disque, énergie : **mesurés** |
| 15 % | Explicabilité | Règles déclenchées, typologie, contributions des features par alerte |
| 10 % | Démo | Rejeu en flux, alertes en direct, rapport généré |
| 10 % | Présentation | 15 min, chiffres issus de `results/` |

## Données (`data/`, jamais committé)
- `data/raw/` via `scripts/get_data.sh` (Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, public) :
  `HI-Small_Trans.csv` (5 078 345 tx, 5 177 illicites, environ 0,1 %), `HI-Small_accounts.csv` (518 581 comptes),
  `HI-Small_Patterns.txt` (370 tentatives, 8 typologies : fan-in, fan-out, cycle, gather-scatter, scatter-gather, bipartite, stack, random).
- `data/prep/` : sorties de préparation déjà générées (`HI-Small_tx.parquet`, `HI-Small_accounts.parquet`,
  `HI-Small_feat.npy` de 894 MB, `HI-Small_meta.json` avec `t0`, taux `usd_per_unit` déduits du dataset et catégories).
  Le code source qui les a produites n'est **pas** dans le dépôt (voir TODO).
- **Pièges** :
  - Le CSV n'est **pas trié** : trier de façon stable par `Timestamp` (résolution à la minute, garder l'ordre du fichier en cas d'égalité).
  - Deux colonnes `Account` : renommer en `from_acct` et `to_acct` dès la lecture. Un compte s'identifie par **(banque, compte)**.
  - Période : 01/09 au 10/09, puis une traîne d'environ 1 100 tx du 11 au 18/09 (fins de tentatives) : la traiter à part.
  - Il existe des auto-virements (`Reinvestment`, émetteur = destinataire) et des paiements multi-devises (Amount Paid ≠ Amount Received).
  - `HI-Small_Patterns.txt` et `Is Laundering` servent **uniquement** à l'évaluation et à l'étiquetage des typologies, jamais comme feature.
- Sources externes (listes GAFI, taux de change, autres variantes IBM) : à documenter dans `data/SOURCES.md` (URL, licence, date, usage) **avant** utilisation. TODO : créer ce fichier.

## Arborescence cible
```
aml/                  paquet Python (seul __init__.py existe aujourd'hui)
  prepare.py          CSV → parquet trié, typé, comptes (banque, compte), conversion en USD       TODO (source perdue, .pyc seulement)
  stream.py           simulateur de rejeu : itère les tx dans l'ordre temporel, vitesse réglable (×N, max), fenêtre de dates
  state.py            état par compte à mémoire bornée : fenêtres glissantes, LRU et TTL, compteurs fixes
  features.py         features incrémentales par compte et motifs de graphe bornés                 TODO (source perdue)
  rules.py            règles explicables → motifs de soupçon rattachés aux 8 typologies
  model.py            chargement et inférence GBDT (LightGBM natif ou ONNX Runtime), contributions par feature
  train.py            entraînement hors ligne sur le split temporel, export du modèle                TODO (source perdue)
  evaluate.py         métriques, choix du seuil sur la validation, baselines → results/
  alerts.py           fusion score, règles et typologie → alerte priorisée et regroupement par compte ou motif
  report/
    tracfin.py        construction du dossier structuré (faits injectés par le code)
    template.md.j2    rapport 100 % template (repli sans LLM)
    narrative.py      narration LLM optionnelle (llama.cpp), qui ne reçoit que des faits déjà calculés
scripts/
  get_data.sh         téléchargement des données (existe)
  bench.py            benchmark de frugalité → results/bench_<machine>.json
  demo.sh             démo de bout en bout
tests/                pytest sur de petits échantillons synthétiques et extraits
results/              métriques et benchmarks VERSIONNÉS (JSON ou CSV + date + commit + machine)
artifacts/            modèles entraînés (.txt LightGBM, .onnx) : non committés, regénérés par train.py
```

## Commandes
```bash
source env.sh                                    # venv Python 3.12, llama.cpp dans le PATH, $MODELS, $LLAMA_ARM64, $QEMU_LD_PREFIX
scripts/get_data.sh                              # HI-Small → data/raw/ (autres variantes : scripts/get_data.sh LI-Small ...)
python -m aml.prepare --variant HI-Small         # TODO : CSV → data/prep/
python -m aml.train --variant HI-Small           # TODO : → artifacts/
python -m aml.evaluate --variant HI-Small        # TODO : → results/metrics.json
python scripts/bench.py --threads 4              # TODO : → results/bench_<host>.json
python -m aml.stream --speed 1000 --from 2022-09-08 --alerts-out alerts.jsonl   # TODO : démo en flux
python -m aml.report --alert <id> [--llm $MODELS/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf]  # TODO : sans --llm, template seul
pytest -q                                        # tests (tâche VS Code « pytest »)
```
LLM (déjà installés) : `llama-server -m $MODELS/<modèle>.gguf -t 4 --port 8080` (CPU x86), `$LLAMA_CUDA` (GPU),
et sur ARM via qemu : `cd $LLAMA_ARM64 && qemu-aarch64 -L $QEMU_LD_PREFIX ./llama-completion -m <gguf> -p "..."`.
GGUF disponibles dans `models/` : Qwen3.5 0.8B, 2B et 4B (Q4_K_M et Q8_0), LFM2.5 350M (Q8_0), LFM2.5 1.2B (Q4_K_M et Q8_0), Gemma-4-E2B (q4_0).
Choix du modèle : à faire **par mesure** (qualité du texte et latence sur la cible) ; prioriser les modèles de 0,35 à 1,2 B pour l'OrangePi.

## Principes d'architecture
1. **Algorithmique d'abord.** Le LLM ne fait que la narration. Il ne produit **jamais** de montant, de date ni d'identifiant :
   le code les injecte (placeholders remplis après génération, ou texte du LLM entre des blocs factuels).
   Vérification après génération : rejeter et retomber sur le template si le texte contient un nombre ou un identifiant absent des faits.
   Le repli 100 % template doit toujours fonctionner (`--no-llm` par défaut dans les tests).
2. **Flux, incrémental, mémoire bornée** (objectif : bien moins de 1 GB de RSS en inférence). Aucun chargement complet du dataset en inférence.
   Lecture en morceaux, état par compte de taille fixe, éviction LRU ou TTL, voisinages de graphe tronqués (degré maximum, profondeur au plus 3, fenêtre temporelle).
3. **Features** par compte sur fenêtres glissantes (1 h, 24 h, 7 j) : nombre, somme et moyenne des montants en USD, degrés entrant et sortant, contreparties distinctes,
   devises, banques et formats distincts, part en espèces, Bitcoin ou ACH, montants ronds, délai de rebond (entrée → sortie),
   ratio sortie/entrée. Motifs bornés : fan-in, fan-out, cycles courts, gather-scatter, scatter-gather.
4. **Modèle** : LightGBM 4.7 (installé ; XGBoost absent), petit nombre d'arbres et de feuilles, export ONNX (TODO : `onnxmltools` absent, sinon `model.txt` natif ou code C).
   Contributions par alerte : `pred_contrib=True` (SHAP pour les arbres, natif LightGBM).
5. **Règles** : chaque règle a un identifiant, un libellé métier en français, la typologie visée et les seuils ; l'alerte liste les règles déclenchées et les 5 premières contributions.
6. **Rapport TRACFIN** : rubriques inspirées des informations publiques de TRACFIN (tracfin.gouv.fr, déclaration de soupçon via ERMES) :
   déclarant (fictif), personnes et comptes concernés, opérations (tableau), motifs du soupçon (typologie, règles, indicateurs),
   analyse narrative, éléments chiffrés et pièces. Citer la source consultée dans le rapport et dans `data/SOURCES.md` ; ne pas inventer d'obligations légales.

## Évaluation (rigueur)
- **Split temporel**, jamais aléatoire. Proposition, à valider puis à consigner dans « Décisions prises » :
  train 01→06/09, val 07→08/09, test 09→10/09 ; traîne 11→18/09 : métriques à part (pas mélangée au test).
- L'état du flux est **causal** : les features d'une tx n'utilisent que les tx antérieures (ordre trié). Le préchauffage sur les jours précédents est permis, mais aucune étiquette n'est utilisée.
- Contrôles anti-fuite (tests) : aucune feature dérivée de `Is Laundering` ou de Patterns ; pas de statistique globale calculée sur val ou test ;
  seuil, hyperparamètres et calibration choisis **sur la validation uniquement**.
- Métriques (classe illicite) : précision, rappel, F1, PR-AUC, précision@k (k = capacité de l'analyste par jour), et
  **taux de tentatives détectées** par typologie (au moins une tx alertée par tentative de Patterns).
- Baselines obligatoires : règles seules ; GBDT sans features de graphe ; GBDT complet.
- Chaque exécution écrit dans `results/` : métriques, seed, commit git, variante des données et date.

## Frugalité (mesurée, pas estimée)
- `scripts/bench.py` : latence par tx p50 et p99 (`time.perf_counter_ns`), débit (tx/s), RSS maximale (`psutil` ou `resource.getrusage`),
  taille sur disque (modèle, état), latence de génération d'un rapport avec et sans LLM, nombre de threads fixé.
- Énergie : RAPL (`/sys/class/powercap/intel-rapl*`) sur x86 si lisible, `tegrastats` sur Jetson, wattmètre externe sur l'OrangePi. Sinon, écrire « non mesuré ».
- Résultats dans `results/bench_<hostname>.json` (versionné). Sous qemu, mesurer la **portabilité** seulement, jamais les performances.

## Conventions de code
- Python 3.12 (vérifié : 3.12.15 dans `.venv`). Le code doit rester compatible avec 3.10 et plus s'il est lancé sur la cible ARM.
- Dépendances d'inférence minimales et disponibles en wheels aarch64 : `numpy`, `pyarrow` ou `polars`, `lightgbm` ou `onnxruntime`, `jinja2`, `psutil`.
  `torch`, `transformers` et CUDA sont **interdits** dans le chemin d'inférence. LLM via le binaire `llama-server` (HTTP) ou `llama_cpp` ; import optionnel avec repli.
  TODO : `requirements.txt` (inférence) et `requirements-dev.txt` (entraînement, tests).
- Pas d'instructions spécifiques à un CPU (pas de `-march=native`, pas d'intrinsèques) ; les accélérations GPU ou NPU restent optionnelles et détectées à l'exécution, avec repli CPU.
- Chemins relatifs à la racine du dépôt (`Path(__file__).resolve().parents[n]`) ou fournis en argument ; aucun chemin absolu.
- Seed fixée (`SEED = 42`) pour numpy, LightGBM et l'échantillonnage. Déterminisme (`deterministic=True`, `num_threads` fixé).
- Lint : `ruff check . && ruff format .` ; types : basedpyright. Montants en `float64`, comparaisons de montants en USD.
- Langue : identifiants en anglais, messages, libellés de règles et rapports en français.

## Règles de travail pour Claude
- **Mesurer avant d'optimiser** ; ne jamais citer un chiffre qui n'a pas été reproduit par un script du dépôt (indiquer le fichier de `results/`).
- Tester d'abord sur un **échantillon** (par exemple 1 jour ou 100 k tx, `--limit`) avant le dataset complet ; surveiller la RAM (`/usr/bin/time -v`).
- Vérifier la portabilité ARM : wheels aarch64 des dépendances, code C ou C++ cross-compilé avec `tools/aarch64.cmake` et exécuté sous `qemu-aarch64`.
- Ne jamais committer `data/`, `models/`, `artifacts/`, `*.gguf`, `*.onnx`, `*.bin`, `.venv/`, `tools/` (déjà dans `.gitignore`, sauf `artifacts/` : TODO l'ajouter).
- Documenter toute source externe (URL, licence, date) avant de l'utiliser.
- Ne pas supprimer ni écraser `data/prep/*` sans le dire. Plusieurs sessions Claude peuvent travailler en parallèle : vérifier `git status` avant d'éditer.
- Pas de commit sans demande explicite. Mettre à jour ce fichier quand une commande passe de TODO à opérationnelle.

## Priorités du hackathon
1. **Bout en bout d'abord** : prepare → stream → features → règles → modèle → alerte → rapport template, même avec des scores faibles.
2. **Scores** : features de graphe, réglage sur la validation, baselines, métriques par typologie.
3. **Frugalité** : bench sur x86 puis sur ARM, réduction (moins d'arbres, ONNX, quantification de l'état).
4. **Démo et présentation** : script de démo, 2 ou 3 rapports exemplaires, slides alimentées par `results/`.

Checklist des livrables :
- [ ] `aml.prepare` (code source restauré) et `data/SOURCES.md`
- [ ] Simulateur de flux `aml.stream` à mémoire bornée
- [ ] Features incrémentales et motifs de graphe bornés
- [ ] Règles rattachées aux 8 typologies
- [ ] Modèle LightGBM entraîné sur le split temporel, exporté (txt ou ONNX)
- [ ] `results/metrics.json` : modèle, baselines, par typologie, traîne 11→18/09
- [ ] Rapport de type TRACFIN : template sans LLM et narration LLM avec garde-fou sur les nombres
- [ ] `scripts/bench.py` et `results/bench_<host>.json` (x86, plus ARM si possible)
- [ ] Démo `scripts/demo.sh` qui tourne sur CPU seul
- [ ] Présentation de 15 min, chaque chiffre relié à un fichier de `results/`

## Décisions prises
Format : `AAAA-MM-JJ : décision. Raison (critère de la grille).`
- 2026-10-26 : sujet AML sur IBM AML HI-Small. Raison : pertinence métier (25 %) et faisable en 2 jours.
- 2026-10-26 : GBDT et règles pour la détection, LLM limité à la narration avec repli template. Raison : frugalité (20 %), explicabilité (15 %).
- 2026-10-26 : LightGBM plutôt que XGBoost (déjà installé, contributions natives, modèle compact). Raison : frugalité, explicabilité.
- 2026-10-26 : llama.cpp compilé en portable (`GGML_NATIVE=OFF`, backends dynamiques) pour x86, CUDA et aarch64. Raison : portabilité imposée.
- À trancher : bornes exactes du split, modèle GGUF retenu, format d'export (txt, ONNX ou C), k de précision@k.
