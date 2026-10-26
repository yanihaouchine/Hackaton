#!/usr/bin/env bash
# Télécharge le dataset IBM AML (Kaggle, public, sans authentification) dans data/raw/.
# Usage : scripts/get_data.sh [VARIANTE ...]   (défaut : HI-Small)
# Variantes : HI-Small LI-Small HI-Medium LI-Medium HI-Large LI-Large
# Attention : les variantes Medium/Large pèsent plusieurs GB.
set -euo pipefail

DATASET="ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
BASE="https://www.kaggle.com/api/v1/datasets/download/$DATASET"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$DEST"

for v in "${@:-HI-Small}"; do
  for f in "${v}_Trans.csv" "${v}_accounts.csv" "${v}_Patterns.txt"; do
    if [[ -s "$DEST/$f" ]]; then
      echo "déjà présent : $f"
      continue
    fi
    echo "téléchargement : $f"
    # -C - : reprend un .part interrompu au lieu de tout retélécharger
    curl -L --fail --retry 3 -C - -o "$DEST/$f.part" "$BASE/$f"
    mv "$DEST/$f.part" "$DEST/$f"
  done
done

ls -lh "$DEST"   # (wc -l prendrait des minutes sur les variantes Large)
