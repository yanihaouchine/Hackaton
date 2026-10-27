#!/usr/bin/env bash
# Mise en place et mesures sur une machine cible (OrangePi Zero 4, Jetson…), sans entraînement.
# Prérequis : python3 (≥ 3.10), gcc ; arbre/ et data/processed/<variante>/ copiés depuis le poste de
# développement, en gardant la structure du dépôt (data/ à la racine, à côté de arbre/), par exemple depuis la racine :
#   rsync -aR --exclude out --exclude build --exclude 'artifacts/models/*/lgbm.txt' \
#         arbre data/processed/HI-Small data/state cible:Hackathon/
# Usage sur la cible : arbre/scripts/target_setup.sh [venv|build|bench|demo]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv-target"
PY="$VENV/bin/python"

step="${1:-all}"
if [[ "$step" == venv || "$step" == all ]]; then
  python3 -m venv "$VENV"
  # inférence : numpy obligatoire ; jinja2/rich/markdown-it pour rapports et démo ; lightgbm pour TreeSHAP (optionnel)
  "$PY" -m pip install -q numpy jinja2 rich markdown-it-py
  "$PY" -m pip install -q lightgbm || echo "lightgbm indisponible : explications par la méthode de Saabas"
fi
if [[ "$step" == build || "$step" == all ]]; then
  CC=gcc "$ROOT/scripts/build_engine.sh" native
fi
if [[ "$step" == bench || "$step" == all ]]; then
  cat <<'TXT'
Mesure d'énergie au wattmètre :
  1. machine au repos 60 s : noter la puissance moyenne P_repos (W) ;
  2. lancer le bench (≈ 1 min sur x86, plus sur ARM) et noter la puissance moyenne pendant le rejeu P_charge ;
  3. relancer : python -m aml bench --watts-idle P_repos --watts-load P_charge
TXT
  OMP_NUM_THREADS=1 "$PY" -m aml bench
  # cœur performant (A76) puis cœur économe (A55) : sur l'OrangePi Zero 4, vérifier la numérotation avec lscpu
  if command -v taskset >/dev/null; then
    for cpu in $(lscpu -p=CPU,MAXMHZ | grep -v '^#' | sort -t, -k2 -n | awk -F, 'NR==1{print $1} END{print $1}'); do
      echo "== cœur $cpu"; OMP_NUM_THREADS=1 taskset -c "$cpu" "$PY" -m aml bench --tag "cpu$cpu"
    done
  fi
fi
if [[ "$step" == demo || "$step" == all ]]; then
  "$PY" -m aml demo --from 2022-09-09 --days 1 --reports 5
fi
