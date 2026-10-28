#!/usr/bin/env bash
# Évalue un modèle figé (entraîné sur HI-Small) sur d'autres variantes entières, sans réentraînement ni réglage.
# Usage : scripts/eval_transfer.sh [VARIANTE...]          (défaut : HI-Medium LI-Small)
#         MODEL=strict NO_CASCADE=1 THREADS=8 MEMMAX=24G scripts/eval_transfer.sh LI-Small
# Pour chaque variante : préparation et features si absentes (data/processed, data/features), puis
# `python -m aml transfer` → results/transfer_<modèle>_<variante>_<horodatage>.json (jamais écrasé).
# Par défaut, configuration livrée (cascade de pré-filtrage), notée en parallèle ; NO_CASCADE=1 note toute
# transaction avec la forêt complète (lent).
# Tout tourne sous plafond mémoire (systemd-run) quand il est disponible ; journaux dans out/logs/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/../.venv/bin/python}"
MODEL="${MODEL:-strict}"
MEMMAX="${MEMMAX:-24G}"
VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(HI-Medium LI-Small)
LOGS="$ROOT/out/logs"
mkdir -p "$LOGS"

capped() {  # commande sous plafond mémoire, sans swap ; repli sans plafond si systemd-run manque
  if command -v systemd-run >/dev/null && systemd-run --user --scope -q true 2>/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$MEMMAX" -p MemorySwapMax=0 "$@"
  else
    echo "  (systemd-run indisponible : pas de plafond mémoire)" >&2
    "$@"
  fi
}

[ -f "build/libamlengine-$(uname -m).so" ] || scripts/build_engine.sh native

for V in "${VARIANTS[@]}"; do
  RAW="$ROOT/../data/raw/${V}_Trans.csv"
  [ -f "$RAW" ] || { echo "$V : $RAW absent (scripts/get_data*.sh à la racine)" >&2; exit 1; }
  LOG="$LOGS/transfer_${MODEL}_${V}_$(date +%Y%m%d-%H%M%S).log"
  echo "== $V (modèle $MODEL) — journal : $LOG"
  {
    if [ ! -f "$ROOT/../data/processed/$V/meta.json" ]; then
      echo "-- préparation"
      capped /usr/bin/time -v "$PY" -m aml prepare --variant "$V"
    fi
    if ! "$PY" -c "import sys; from aml.config import EngineConfig; from aml.features import feature_dir; \
sys.exit(not (feature_dir('$V', EngineConfig()) / 'meta.json').exists())"; then
      echo "-- features"
      capped /usr/bin/time -v "$PY" -m aml features --variant "$V"
    fi
    echo "-- évaluation"
    capped /usr/bin/time -v "$PY" -m aml transfer --variant "$V" --model "$MODEL" ${NO_CASCADE:+--no-cascade} ${THREADS:+--threads "$THREADS"}
  } 2>&1 | tee "$LOG"
done
