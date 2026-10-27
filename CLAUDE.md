# CLAUDE.md : hackathon AML basse consommation

Référence de toutes les sessions Claude Code sur ce dépôt. À tenir à jour, surtout « Décisions prises ».
Le dépôt accueille plusieurs solutions, chacune dans son dossier avec son propre `CLAUDE.md` (architecture, décisions, chiffres) :
- `arbre/` : moteur de features en C et forêt d'arbres (LightGBM) ; voir `arbre/CLAUDE.md`.
- la solution par réseaux de neurones de l'équipe, dans son propre dossier.

Ce fichier ne contient que ce qui est commun : sujet, grille, contraintes, données, principes et règles de travail.
Ne pas modifier le dossier d'une autre solution sans le dire. Toute évolution d'architecture se propose avant d'être codée.

## Projet
Lutte anti-blanchiment (LCB-FT) : à partir du flux de transactions IBM AML, lever des alertes **justifiées** et produire pour chacune
un rapport inspiré de la déclaration de soupçon **TRACFIN**. Hackathon de 2 jours (26 et 27/10/2026).
Les organisateurs n'ont fourni que le dataset : le rejeu en flux et le modèle de rapport sont à construire nous-mêmes.
Les données publiques externes sont autorisées.

Grille de notation : chaque choix doit se rattacher à au moins une ligne.
| Poids | Critère | Attendu |
|---|---|---|
| 25 % | Pertinence métier | Utile à un analyste conformité : alertes priorisées, typologie nommée, rapport exploitable |
| 20 % | Qualité technique | Résultats mesurés et bien évalués |
| 20 % | Frugalité | Taille, énergie, latence |
| 15 % | Explicabilité | Chaque alerte se justifie |
| 10 % | Démonstration | Elle fonctionne et se comprend |
| 10 % | Présentation | Claire, tenue en 15 minutes |

## Contraintes matérielles
- Cible : machines de 4 à 7 W. La plus faible est l'**OrangePi Zero 4** (4 GB de RAM, 2× A76 et 6× A55, NPU de 3 TOPS).
  Autres machines de test : Jetson Orin (ARM, 32 GB), RTX 3060 12 GB (30 GB de RAM), Xeon 56 cœurs avec 2× A6000.
- Code **indépendant de l'architecture** (Intel, AMD, ARM64) : pas de `-march=native`, pas d'intrinsèques, pas de dépendance obligatoire à CUDA ou au NPU.
  Toute accélération reste optionnelle, détectée à l'exécution, avec un repli CPU.

## Données (`data/` à la racine, partagé entre les solutions, jamais committé)
- `scripts/get_data.sh [VARIANTE...]` télécharge dans `data/raw/` depuis Kaggle (`ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, public, sans clé).
  La variante par défaut est HI-Small. `scripts/get_data_large.sh` récupère les autres (LI-Small, HI/LI-Medium, HI/LI-Large : environ 38 GiB),
  après avoir vérifié l'espace disque ; les téléchargements interrompus reprennent. Ne jamais charger une variante Large entièrement en RAM.
- HI-Small (vérifié) :
  - `HI-Small_Trans.csv` : 5 078 345 transactions, dont 5 177 illicites (0,102 %).
  - `HI-Small_accounts.csv` : 518 581 comptes (banque, compte, entité).
  - `HI-Small_Patterns.txt` : 370 tentatives en 8 typologies (fan-in, fan-out, cycle, gather-scatter, scatter-gather, bipartite, stack, random).
- Formats de paiement : Cheque, Credit Card, ACH, Cash, Reinvestment, Wire, Bitcoin.
  591 212 auto-virements (émetteur = destinataire) ; 72 170 transactions où la devise payée diffère de la devise reçue.
- **Pièges** :
  - Le CSV n'est **pas trié** par date : faire un tri stable sur `Timestamp` (résolution à la minute).
  - Deux colonnes portent le nom `Account` (émetteur et destinataire) : les renommer dès la lecture. Un compte s'identifie par le couple **(banque, compte)**.
  - Les codes banque ont des zéros en tête (`010`, `03208`) : les lire en **chaîne**, pas en entier.
  - La période couvre surtout le 01/09 au 10/09 ; il reste une traîne d'environ 1 100 transactions du 11 au 18/09, fins de tentatives probables. La traiter à part.
  - La traîne est à environ 60 % illicite : la date seule y « prédit » l'étiquette (fuite si elle est mélangée au test).
  - `Patterns.txt` ne couvre que 3 209 des 5 177 transactions illicites (62 %) : le reste n'a pas de typologie. Mesurer le rappel aussi hors motifs.
  - Les codes banque de `accounts.csv` (`Bank ID`) n'ont pas de zéros en tête, contrairement à ceux de `Trans.csv` : normaliser avant toute jointure.
  - `Is Laundering` et `Patterns.txt` sont des **étiquettes** : elles servent à l'entraînement et à l'évaluation, jamais comme feature.
- Tout fichier dérivé (sous `data/`, hors `raw/`) doit être régénérable par un script du dépôt. Sous-dossiers déjà utilisés par `arbre/` :
  `data/processed/`, `data/features/`, `data/state/` ; une autre solution prend ses propres sous-dossiers.
- Toute source externe (listes GAFI, taux de change…) doit être documentée (URL, licence, date, usage) dans `data/SOURCES.md` **avant** utilisation.

## Principes non négociables
1. **Détection 100 % algorithmique**, sans LLM (statistiques, graphe, règles, modèle classique).
2. **Le LLM a deux rôles, et seulement deux** : rédiger le rapport TRACFIN, et l'assistant d'analyse qui répond aux questions de l'analyste sur une alerte.
   - Il ne reçoit que des faits déjà calculés et ne répond qu'à partir du dossier de l'alerte.
   - Il n'écrit **jamais** un montant, une date ou un identifiant : le code les injecte. Tout texte généré qui contient un nombre absent des faits est rejeté.
   - Un repli **sans LLM** (templates) doit toujours fonctionner.
3. **Flux causal** : le traitement d'une transaction n'utilise que les transactions antérieures. Aucune information du futur, aucune étiquette.
4. **Mémoire bornée** en inférence : rien qui charge tout le dataset, un état de taille contrôlée, un objectif de RAM compatible avec 4 GB (et bien en dessous).
5. **Chaque alerte se justifie** : raisons lisibles par un humain (règles, indicateurs, typologie), pas seulement un score.

## Rigueur d'évaluation
- Découpage **temporel** (train, validation, test), jamais aléatoire ; traîne du 11 au 18/09 évaluée à part. Bornes : voir « Décisions prises ».
- Seuil, hyperparamètres et calibration sont choisis **sur la validation uniquement** ; le test n'est regardé qu'une fois la configuration figée.
- Métriques sur la classe illicite : précision, rappel, F1, PR-AUC, précision@k (k = capacité quotidienne d'un analyste),
  et taux de tentatives détectées **par typologie**.
- Toujours comparer à des baselines (au minimum : règles seules, modèle sans features de graphe).
- Tester l'absence de fuite : aucune feature dérivée des étiquettes, aucune statistique calculée sur la validation ou le test.
- Tout chiffre cité (README, slides, conversation) provient d'un script relançable qui l'écrit dans un fichier versionné, avec la date, le commit, la seed et la machine.

## Frugalité : mesurée, jamais estimée
- Mesures : latence par transaction (p50 et p99), débit, RAM maximale (RSS), taille sur disque (modèle et état), latence d'un rapport avec et sans LLM ; nombre de threads fixé.
- Énergie : RAPL sur x86 si lisible, `tegrastats` sur Jetson, wattmètre sur l'OrangePi ; sinon écrire « non mesuré ».
- Sous qemu, on vérifie la **portabilité**, jamais les performances.

## Conventions
- Python compatible 3.10 et plus. Dépendances d'inférence minimales et disponibles en wheels aarch64.
- Chemins relatifs à la racine du dépôt (ou du dossier de la solution) ou passés en argument ; aucun chemin absolu.
- Seed fixée (42) et exécutions déterministes (threads fixés).
- Identifiants de code en anglais ; libellés d'alerte, rapports et messages en français.
- `ruff check . && ruff format .` (depuis le dossier de la solution) avant de proposer un commit.

## Règles de travail pour Claude
- **Ne pas coder sans validation** de l'approche par l'utilisateur ; proposer d'abord, brièvement, en rattachant la proposition à la grille.
- Mesurer avant d'optimiser. Ne jamais annoncer un résultat non reproduit ; si un test échoue ou une étape est sautée, le dire.
- Travailler d'abord sur un **échantillon** (un jour, ou environ 100 k transactions), puis sur le dataset complet ; surveiller la RAM (`/usr/bin/time -v`).
- Vérifier la portabilité ARM dès qu'une dépendance ou du code natif est ajouté (outils qemu/aarch64 à réinstaller au besoin : voir « Environnement local »).
- Ne jamais committer `data/`, `models/`, `tools/`, `.venv/`, `env.sh`, `.vscode/`, `.env`, `*.gguf`, `*.onnx`, `*.bin` (voir `.gitignore`).
  Exceptions : `arbre/artifacts/*.onnx` et les fichiers d'inférence des modèles livrés de `arbre/artifacts/models/` (liste dans `.gitignore`).
- Pas de commit ni de push sans demande explicite. Vérifier `git status` avant d'éditer : d'autres sessions peuvent travailler en parallèle.
- Ne rien supprimer sous `data/` sans le dire.
- Ne pas lancer de commande qui réécrit un fichier de `results/` (démo, bench…) juste pour tester : il est cité ailleurs.

## Environnement local (racine du dépôt, partagé)
- `.venv/` : Python 3.12 (polars, lightgbm, numpy, jinja2, rich, llama_cpp_python) ; `source env.sh`.
- `models/` : LLM GGUF locaux (qwen3.5-0.8b/2b/4b, lfm2.5-350m/1.2b, gemma-4-e2b), utilisés pour les rapports.
- qemu-aarch64 sans droits root : `cd tools/qemu && apt-get download qemu-user && dpkg -x qemu-user_*.deb extracted` (binaire statique `tools/qemu/extracted/usr/bin/qemu-aarch64`).
- Tâches longues ou gourmandes (bench LLM, entraînement) : les lancer sous plafond mémoire
  (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 …`) et écrire les logs hors de `/tmp`. Le PC a gelé le 2026-10-27 pendant un bench LLM lancé sans plafond.

## Décisions prises (communes)
Format : `AAAA-MM-JJ : décision. Raison (critère de la grille).`
- 2026-10-26 : découpage temporel : train 01–06/09, validation 07–08/09, test 09–10/09, traîne 11–18/09 évaluée à part ; « protocole littérature » (test du 09/09 à la fin, traîne comprise) rapporté séparément. C'est le découpage par jours de Multi-GNN (60/20/20), recalculé. Raison : comparabilité avec l'état de l'art sans mélanger la traîne au test, et entre les solutions du dépôt (qualité technique).
- 2026-10-26 : `data/SOURCES.md` versionné grâce à `data/*` + `!data/SOURCES.md` dans `.gitignore` (sinon jamais committé).
- 2026-10-27 : dépôt découpé par solution : `arbre/` pour la forêt d'arbres, un autre dossier pour les réseaux de neurones ; `data/`, `models/`, `.venv/`, `tools/` et `scripts/get_data*.sh` restent à la racine, partagés. Raison : deux équipes poussent dans le même dépôt (présentation, démo).
