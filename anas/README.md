# Solution d'Anas : détection en flux + rapport inspiré de la DS TRACFIN

Pipeline complet, sans LLM : données → détection par règles en flux (causale, mémoire bornée)
→ alertes priorisées → dossier de faits → rapport en français.

## Installation (une fois)

```bash
cd ~/Hackaton                       # racine du dépôt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r anas/requirements-report.txt
```

Il n'y a rien à compiler : c'est du Python pur (polars, pyarrow, jinja2).
Les données brutes doivent être dans `data/raw/` (`scripts/get_data.sh`).

## Exécution (depuis `anas/`, avec le `.venv` activé)

```bash
cd anas
python scripts/prepare_data.py              # 1. tri du flux, rapprochement Patterns, taux internes (~5 s)
python scripts/tune.py                      # 2. seuils choisis sur la validation -> config/thresholds.json (~40 s)
python scripts/detect.py --split test       # 3. rejeu du flux du 08 au 10/09, alertes (~45 s)
python scripts/evaluate.py --split test     # 4. métriques -> results/eval_test.json
python scripts/report.py --split test       # 5. 20 rapports -> ../data/anas/reports/index.md
python -m pytest -q                         # tests
```

Rapport immédiat, sans données : `python scripts/report.py --example`
(→ `../data/anas/reports/exemple_fanout.md`). Essai rapide sur un échantillon :
`python scripts/detect.py --split val --hours 6`.

`config/thresholds.json` est versionné : l'étape 2 est facultative.

## Organisation

| Fichier | Rôle |
|---|---|
| `src/aml/data.py` | lecture du CSV (colonnes renommées, codes banque en chaîne, tri stable), Patterns.txt |
| `src/aml/engine.py` | moteur en flux : fenêtres de 24 h par compte, règles, alertes |
| `src/aml/stream.py` | rejeu par lots du parquet (les étiquettes ne sont jamais lues) |
| `src/aml/metrics.py` | précision, rappel, F1, PR-AUC, précision@k, tentatives par typologie |
| `src/aml/dossier.py` | faits numérotés F1, F2… calculés une seule fois, formatés en français |
| `src/aml/writers.py` | rédaction : templates (toujours dispo) + emplacement LLM avec repli |
| `src/aml/guard.py` | garde-fou : tout nombre rédigé doit exister dans les faits |
| `src/aml/schema.py` | contrat `Alert` / `AlertDossier` (exemple : `contracts/examples/`) |
| `templates/rapport_fr.md.j2` | modèle du rapport (8 sections) |

## Règles (typologies du kick-off)

Dispersion (fan-out), rassemblement (fan-in), rassemblement puis dispersion, chaîne rapide,
circuit fermé, fractionnement, compte réveillé. Les règles ne se déclenchent que sur les formats
de paiement risqués, fixés sur l'entraînement (ACH). Découpage : entraînement 01–05/09,
validation 06–07/09, test 08–10/09 ; traîne 11–18/09 non utilisée.

## Résultats

Tous les chiffres sont dans `results/*.json`, avec la date, le commit, la seed et la machine.
Les principaux : `results/eval_test.json` (qualité), `results/detect_test.json` (débit, latence,
RAM), `results/report_bench_template.json` (latence d'un rapport).

Limites connues : la précision des règles seules est faible (beaucoup d'alertes par jour),
ce qui donne une précision@k basse. Le classement des alertes par un modèle est la prochaine étape.
