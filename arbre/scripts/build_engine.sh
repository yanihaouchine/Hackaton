#!/usr/bin/env bash
# Compile le moteur C (aml/engine/aml_engine.c) en bibliothèque partagée, sans -march (code portable).
# Usage : scripts/build_engine.sh            → build/libamlengine-<arch>.so pour la machine courante (cc)
#         scripts/build_engine.sh aarch64    → build/libamlengine-aarch64.so en compilation croisée (zig cc)
#         scripts/build_engine.sh x86_64     → build/libamlengine-x86_64.so, cible de base x86-64 (zig cc)
# zig est fourni par le paquet Python « ziglang » (python -m ziglang) : aucune chaîne croisée à installer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/aml/engine/aml_engine.c"
OUT="$ROOT/build"
mkdir -p "$OUT"
FLAGS=(-O2 -std=c99 -ffp-contract=off -fPIC -shared -s)  # pas de FMA implicite : mêmes flottants partout
PY="${PYTHON:-$ROOT/../.venv/bin/python}"

case "${1:-native}" in
  native)
    "${CC:-cc}" "${FLAGS[@]}" -o "$OUT/libamlengine-$(uname -m).so" "$SRC" -lm ;;
  aarch64|x86_64)
    # -mcpu=baseline : jeu d'instructions de base de l'architecture (jamais celui de la machine de build)
    "$PY" -m ziglang cc -target "$1-linux-gnu.2.17" -mcpu=baseline "${FLAGS[@]}" \
      -o "$OUT/libamlengine-$1.so" "$SRC" -lm ;;
  *) echo "cible inconnue : $1" >&2; exit 1 ;;
esac
ls -l "$OUT"/libamlengine-*.so
