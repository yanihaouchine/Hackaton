# CLAUDE.md : solution par arbres (`arbre/`)

Complète le `CLAUDE.md` de la racine (sujet, grille, données, principes, règles de travail), qui s'applique aussi ici.
État actuel : architecture validée le 2026-10-26 (voir « Décisions prises »), code dans `aml/` (point d'entrée `python -m aml`, à lancer depuis `arbre/`),
moteur de features en C dans `aml/engine/`, tests dans `tests/`, chiffres dans `results/`, modèles livrés dans `artifacts/models/`.

## Chemins
- `aml/config.py` est la source unique des chemins : `ROOT` = `arbre/` (code, `results/`, `artifacts/`, `build/`, `out/`),
  `REPO` = racine du dépôt (ressources partagées : `data/`, `models/`, `tools/`, `.venv/`).
- Python : `../.venv/bin/python` depuis `arbre/` (ou `source ../env.sh`).

## Environnement local
- Compilation du moteur : `scripts/build_engine.sh [native|aarch64]` ; aarch64 en croisé sans chaîne dédiée via `python -m ziglang cc -target aarch64-linux-gnu -mcpu=baseline` (paquet pip `ziglang`).
- `tests/test_portability.py` utilise le qemu-aarch64 de `../tools/qemu/` (installation : voir le `CLAUDE.md` de la racine).
- Cible : `scripts/target_setup.sh` (venv minimal, compilation native, bench avec wattmètre, démo).

## Décisions prises
Format : `AAAA-MM-JJ : décision. Raison (critère de la grille).` Le découpage temporel est consigné dans le `CLAUDE.md` de la racine.
- 2026-10-26 : architecture : préparation hors ligne (polars) → moteur de features en C99 portable appelé par ctypes (`aml/engine/`), le même code pour l'entraînement et l'inférence → forêt LightGBM aplatie, évaluée en C → cas (union-find) → rapport par template, LLM optionnel sous garde. Raison : frugalité (≈ 4,4·10⁵ tx/s sur un cœur x86, état ≈ 160 Mo) et aucun écart entraînement/inférence (qualité technique).
- 2026-10-26 : compilation sans `-march`, avec `-ffp-contract=off`, `-std=c99` ; aarch64 en croisé via `python -m ziglang cc -target aarch64-linux-gnu -mcpu=baseline` (`scripts/build_engine.sh`). Raison : portabilité et flottants identiques d'une architecture à l'autre (frugalité, démo).
- 2026-10-26 : deux modes : **strict** (score à l'arrivée, métrique principale) et **revue J+1** (re-score à minuit avec les seules données antérieures à la décision ; métrique secondaire, étiquetée). Raison : principe 3 respecté ; comparaison honnête avec les GNN qui voient tout le graphe de test (qualité technique, pertinence métier).
- 2026-10-26 : graphe de motifs restreint à l'ACH (3 208 des 3 209 transactions de motifs de HI-Small) ; fenêtres 6 h / 24 h / 96 h (durée médiane des tentatives : 24 à 150 h). Raison : frugalité (≈ 0,53 M arêtes au lieu de 4,5 M) ; ablation « tous formats » à mesurer.
- 2026-10-26 : taux de change dérivés des transactions multi-devises du train (médiane USD→X ; résidu médian 4·10⁻⁷), aucune source externe. Raison : le simulateur utilise des taux fixes ; reproductibilité.
- 2026-10-26 : entraînement avec toutes les illicites et 25 % des licites ; probabilité recalibrée analytiquement (cote × 0,25) ; seuil = F1 maximal sur la validation ; 5 seeds pour la variance. Raison : entraînement 4 fois plus rapide ; comparaison avec 100 % des licites non encore menée (recherche interrompue, machine saturée) (qualité technique).
- 2026-10-27 : cascade de pré-filtrage (60 arbres) devant la forêt complète, seuil fixé sur la validation pour conserver 99 % des alertes ; mesuré : débit ×33,6 sur x86 (6 366 → 214 055 tx/s), F1 test −0,005. Raison : frugalité (`results/val_cascade_strict.json`, `results/bench_*`).
- 2026-10-27 : modèle temps réel livré = « strict + cascade ». L'étape 2 (propagation des scores, la plus précise : F1 test 0,662 contre 0,653) est rapportée comme configuration de précision. Ses scores propagés ont été appris sans cascade, elle doit donc tourner sans cascade, avec deux forêts complètes par transaction (débit non mesuré ; une seule forêt sans cascade : 6 366 tx/s sur x86). Raison : compromis frugalité/précision explicite (frugalité, qualité technique).
- 2026-10-27 : la revue J+1 n'apporte rien sur le test propre (F1 0,653). Elle est gardée dans la démo comme complément du lendemain (alertes nouvelles à minuit), sans être présentée comme un gain de détection. Raison : honnêteté des résultats (qualité technique).
- 2026-10-27 : versionnés dans `artifacts/models/` : les fichiers d'inférence de `strict` (avec la cascade), `typology` et `j1` (chargé par la démo) ; `lgbm.txt` et les variantes d'expérience restent locaux. Raison : la démo tourne depuis un clone sans réentraînement, pour ≈ 4 Mo (démonstration, frugalité).
- 2026-10-27 : `lgbm.txt` de `strict` et `j1` désormais versionné (remplace l'exclusion ci-dessus), régénéré par `python -m aml export-lgbm`, qui réentraîne la seed livrée et n'écrit le fichier que si la forêt a l'empreinte livrée ; le repli Saabas se déclenche aussi quand le fichier manque (il plantait). Raison : explications TreeSHAP exactes sur toutes les machines, sans réentraînement (explicabilité, démonstration).
