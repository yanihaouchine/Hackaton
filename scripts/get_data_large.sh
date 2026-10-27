#!/usr/bin/env bash
# Télécharge les variantes plus grosses du dataset IBM AML dans data/raw/ (via get_data.sh).
# Usage : scripts/get_data_large.sh [VARIANTE ...]
# Défaut : LI-Small HI-Medium LI-Medium HI-Large LI-Large (environ 41 GB au total)
# Tailles indicatives (Trans.csv) : LI-Small 0,65 GB · *-Medium 3 GB · *-Large 17 GB.
# Relançable : les fichiers complets sont sautés, les téléchargements interrompus reprennent.
set -euo pipefail

DATASET="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
BASE="https://www.kaggle.com/api/v1/datasets/download/$DATASET"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/raw"
mkdir -p "$DEST"

if [[ $# -gt 0 ]]; then
  VARIANTS=("$@")
else
  VARIANTS=(LI-Small HI-Medium LI-Medium HI-Large LI-Large)
fi

# Taille restant à télécharger (lue sur le serveur), comparée à l'espace libre
need=0
for v in "${VARIANTS[@]}"; do
  for f in "${v}_Trans.csv" "${v}_accounts.csv" "${v}_Patterns.txt"; do
    [[ -s "$DEST/$f" ]] && continue
    size=$(curl -sSL -r 0-0 -o /dev/null -D - "$BASE/$f" | tr -d '\r' | awk -F/ 'tolower($0) ~ /^content-range/ {s=$NF} END {print s+0}')
    [[ "$size" -gt 0 ]] || { echo "introuvable sur Kaggle : $f" >&2; exit 1; }
    have=$(stat -c %s "$DEST/$f.part" 2>/dev/null || echo 0)
    need=$((need + size - have))
  done
done
free=$(( $(df -Pk "$DEST" | awk 'NR==2 {print $4}') * 1024 ))
echo "à télécharger : $((need / 1024**3)) GB · libre : $((free / 1024**3)) GB"
if [[ "$need" -gt "$free" ]]; then
  echo "espace disque insuffisant" >&2
  exit 1
fi

"$ROOT/scripts/get_data.sh" "${VARIANTS[@]}"
