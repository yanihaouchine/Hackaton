#!/usr/bin/env bash
# Lance l'interface de suivi des alertes en une commande, depuis n'importe où dans le dépôt :
#   bash neurones/interface/run.sh            → http://localhost:8501
# Si aucune alerte n'a encore été produite, rejoue d'abord la journée du 09/09 (≈ 15 s, features requises).
# Variables : PORT (8501 par défaut), PYTHON (../.venv/bin/python par défaut).
set -euo pipefail
NEURONES="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$NEURONES/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
cd "$NEURONES"
if ! ls "$REPO"/data/neurones/alertes/*/index.json >/dev/null 2>&1; then
  echo "Aucune alerte trouvée : rejeu de la journée du 09/09 avec la détection de arbre/…"
  "$PY" -m rapport.make_alerts
fi
exec "$PY" -m streamlit run interface/app.py --server.port "${PORT:-8501}" --server.headless true \
  --browser.gatherUsageStats false
