# CLAUDE.md : solution par réseau de neurones (`neurones/`)

Complète le `CLAUDE.md` de la racine (sujet, grille, données, principes, règles de travail), qui s'applique aussi ici.
État actuel : MLP entraîné, exporté en ONNX et évalué une fois sur le test ; présentation et commandes dans `README.md`.
Code dans `nn/`, scripts dans `scripts/`, chiffres dans `results/`, modèles régénérables dans `artifacts/` (hors git).

## Chemins
- `nn/data.py` ajoute `arbre/` au `sys.path` et réutilise son code en lecture seule (flux, étiquettes, features, découpage, tirage, métriques).
- Python : `../.venv/bin/python` depuis `neurones/`.

## Décisions prises
Format : `AAAA-MM-JJ : décision. Raison (critère de la grille).` Le découpage temporel est consigné dans le `CLAUDE.md` de la racine.
- 2026-10-27 : le MLP reprend les 96 features du modèle strict de `arbre/` (moteur C, empreinte `03a26672a4ef`), son découpage et son tirage (toutes les illicites + 25 % des licites, seed 42), en important son code. Raison : comparaison forêt / réseau sans biais (qualité technique).
- 2026-10-27 : normalisation : one-hot pour les 8 codes sans ordre, signe·log(1+|x|) puis centrage-réduction (bornée à ±10) pour les autres ; statistiques du train uniquement ; normalisation intégrée au modèle exporté. Raison : un réseau exige des entrées à la même échelle ; aucune fuite (qualité technique).
- 2026-10-27 : architecture 220 → 128 → 64 → 1 (ReLU, dropout 0,1, 36 609 paramètres), Adam lr 1e-3, lots de 4 096, 4 threads. Raison : le plus petit réseau raisonnable (frugalité).
- 2026-10-27 : arrêt précoce sur la loss de validation (patience 3) ; poids des fraudes choisi parmi 1, 10, 100 sur la PR-AUC de validation (la loss n'est pas comparable d'un poids à l'autre) : retenu 1 ; seuil = F1 maximal sur la validation ; 3 seeds pour la variance, modèle livré = seed 42. Raison : réglages sur la validation uniquement (qualité technique).
- 2026-10-27 : livrable en ONNX (onnxruntime), torch réservé à l'entraînement ; écart torch / ONNX ≤ 1,4·10⁻⁷. Raison : dépendances d'inférence minimales et portables (frugalité, `CLAUDE.md`).
- 2026-10-27 : explications par gradient × entrée (one-hot regroupés), au format des raisons de `arbre/` ; fidélité mesurée par le protocole de `arbre/`. Raison : TreeSHAP ne s'applique pas à un réseau (explicabilité).
- 2026-10-27 : test évalué une fois (`results/eval_mlp_test.json`) : F1 0,606 contre 0,653 pour la forêt strict. La forêt reste le modèle de détection ; le MLP est présenté comme comparaison mesurée (plus petit : 156 Ko contre 1,6 Mo ; moins précis ; moins rapide que la forêt avec cascade). Raison : honnêteté des résultats (qualité technique, frugalité).
