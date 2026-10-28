#!/usr/bin/env bash
# Démo devant le jury : rejeu d'une tranche d'un jour de TEST par le mélange forêt + modèle séquentiel,
# puis l'interface d'Anas (limitée à cette machine). Ne modifie ni arbre/, ni rapport/, ni interface/.
#
#   bash neurones/scripts/demo.sh verifier                      contrôle que tout est prêt
#   bash neurones/scripts/demo.sh tranche 2022-09-09 08:00 3    rejoue le 09/09 de 8 h à 11 h (jour, heure, durée en h)
#   bash neurones/scripts/demo.sh interface                     lance l'interface sur http://localhost:8501
#   bash neurones/scripts/demo.sh verite 2022-09-09 08:00       vérité de la tranche (à montrer après coup)
#   bash neurones/scripts/demo.sh secours                       prépare 4 tranches à l'avance (plan B)
#
# Variables : PORT (8501), PYTHON (.venv de la racine).
set -euo pipefail
NEURONES="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$NEURONES/.." && pwd)"
PY="${PYTHON:-$REPO/.venv/bin/python}"
PORT="${PORT:-8501}"
ALERTS="$REPO/data/neurones/alertes"
cd "$NEURONES"
export PYTHONWARNINGS=ignore

ok() { printf '  [ok]   %s\n' "$1"; }
ko() { printf '  [KO]   %s\n' "$1"; FAILED=1; }

verifier() {
  FAILED=0
  echo "Vérification de la démo"
  [[ -x "$PY" ]] && ok "Python : $PY" || ko "Python introuvable ($PY) : créer le .venv de la racine"
  for m in numpy torch onnxruntime streamlit jinja2 polars; do
    "$PY" -c "import $m" 2>/dev/null && ok "paquet $m" || ko "paquet $m manquant : $PY -m pip install $m"
  done
  ls "$REPO"/arbre/build/libamlengine-*.so >/dev/null 2>&1 && ok "moteur C compilé" \
    || ko "moteur C absent : (cd arbre && bash scripts/build_engine.sh)"
  [[ -f "$REPO/data/processed/HI-Small/meta.json" ]] && ok "flux préparé" \
    || ko "flux absent : (cd arbre && ../.venv/bin/python -m aml prepare)"
  [[ -f "$REPO/data/features/HI-Small/03a26672a4ef/X.npy" ]] && ok "features du moteur" \
    || ko "features absentes : (cd arbre && ../.venv/bin/python -m aml features)"
  [[ -f "$NEURONES/artifacts/hist_h64-32_g16_s42/model.pt" ]] && ok "modèle séquentiel livré" \
    || ko "modèle séquentiel absent : (cd neurones && ../.venv/bin/python -m mlp.train_hist --seed 42)"
  [[ -f "$NEURONES/results/val_blend_hist.json" ]] && ok "mélange figé (poids et seuil)" \
    || ko "mélange absent : (cd neurones && ../.venv/bin/python -m mlp.blend_hist)"
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    ok "port $PORT déjà utilisé : l'interface tourne sans doute déjà (http://localhost:$PORT)"
  else
    ok "port $PORT libre"
  fi
  [[ "$FAILED" == 0 ]] && echo "Tout est prêt." || { echo "Des éléments manquent (voir [KO])."; exit 1; }
}

tranche() {
  local date="${1:?jour : 2022-09-09 ou 2022-09-10}" start="${2:-08:00}" hours="${3:-3}"
  echo "Rejeu du $date à partir de $start, pendant $hours h (mélange forêt + séquentiel)…"
  "$PY" -m mlp.make_alerts_hist --date "$date" --start "$start" --hours "$hours" 2>&1 \
    | grep -v -e "^vérité" -e RuntimeWarning
  echo "Dans l'interface, choisir la journée « ${date}_${start/:/h}_melange »."
}

interface() {
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "L'interface tourne déjà : http://localhost:$PORT"
    return
  fi
  ls "$ALERTS"/*/index.json >/dev/null 2>&1 || { echo "Aucune tranche : lancer d'abord « demo.sh tranche »."; exit 1; }
  echo "Interface sur http://localhost:$PORT (cette machine seulement ; Ctrl+C pour arrêter)"
  exec "$PY" -m streamlit run interface/app.py --server.port "$PORT" --server.address 127.0.0.1 \
    --server.headless true --browser.gatherUsageStats false
}

verite() {
  local date="${1:?jour}" start="${2:-08:00}"
  local f="$ALERTS/${date}_${start/:/h}_melange/truth.json"
  [[ -f "$f" ]] || { echo "Pas de vérité pour cette tranche : lancer d'abord « demo.sh tranche $date $start »."; exit 1; }
  "$PY" - "$f" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))
print(f"Fraudes réelles dans la tranche : {t['n_illicit']}")
print(f"  attrapées             : {t['caught']}")
print(f"  ratées                : {t['missed']}  (dont {t['missed_non_pattern']} hors motifs, l'angle mort)")
print(f"Fausses alertes         : {t['false_alerts']}")
print(f"Tentatives détectées    : {t['attempts_detected']} / {t['attempts']}")
print(f"Cas contenant une fraude : {t['true_cases']}")
EOF
}

secours() {
  for spec in "2022-09-09 08:00 3" "2022-09-09 14:00 3" "2022-09-10 10:00 3" "2022-09-10 18:00 3"; do
    # shellcheck disable=SC2086
    tranche $spec
  done
  echo "Tranches de secours prêtes ; elles apparaissent toutes dans l'interface."
}

case "${1:-}" in
  verifier) verifier ;;
  tranche) shift; tranche "$@" ;;
  interface) interface ;;
  verite) shift; verite "$@" ;;
  secours) secours ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
