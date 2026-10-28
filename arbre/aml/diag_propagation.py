"""Diagnostic : parmi les transactions illicites que le modèle manque, combien une propagation de suspicion
entre comptes aurait-elle pu signaler ? Lecture seule ; écrit out/diag/diag_propagation_<horodatage>.json.

Pour chaque transaction de la validation, on ne regarde que le passé (transactions antérieures) et on mesure
la distance, en nombre de « sauts » dans le réseau des échanges, jusqu'au plus proche compte déjà marqué :
  0 : l'émetteur ou le destinataire a lui-même déjà participé à une transaction marquée ;
  1 : un compte avec qui l'un d'eux a déjà échangé est marqué ;
  2 : un voisin de voisin est marqué ; etc. jusqu'à --max-hops ;
  isolé : aucun compte marqué à portée.
Deux définitions de « transaction marquée » :
  alertes   : alertes réellement levées, chaque jour noté par un modèle qui ne l'a pas vu à l'entraînement
              (jours 0–2 : strict_b, appris sur 3–5 ; jours 3–5 : strict_a, appris sur 0–2 ; validation : strict) ;
  etiquettes : toutes les illicites passées, comme si chacune avait été repérée (plafond). Les étiquettes ne
              servent qu'à ce diagnostic, jamais au modèle.
Plafond optimiste : un chemin compte même si les échanges qui le composent ont eu lieu avant le marquage.
Pour mesurer le prix en fausses alertes, on classe aussi un échantillon de transactions licites.

Lancement (depuis arbre/) : ../.venv/bin/python -m aml.diag_propagation [--windows 0 96] [--max-rows N]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

from aml.config import ROOT
from aml.data import load, load_labels
from aml.features import load_features
from aml.forest import c_raw
from aml.model import load_bundle
from aml.provenance import run_info

SEED = 42


def _alerts(name: str, X, lo: int, hi: int, chunk: int = 200_000) -> np.ndarray:
    b = load_bundle(name)
    out = np.empty(hi - lo, dtype=bool)
    for a in range(lo, hi, chunk):
        z = min(a + chunk, hi)
        p = 1.0 / (1.0 + np.exp(-c_raw(b.forest, np.asarray(X[a:z], dtype=np.float32))))
        out[a - lo : z - lo] = p >= b.threshold
        print(f"  {name} : {z - lo}/{hi - lo}", flush=True)
    return out


class History:
    """Index (compte, ligne) triés : voisins et marquages d'un compte sur un intervalle de lignes [lo, i)."""

    def __init__(self, src: np.ndarray, dst: np.ndarray, marked: np.ndarray):
        n = len(src)
        self.big = np.int64(n + 1)
        keep = src != dst  # auto-virements : pas de voisin
        rows = np.flatnonzero(keep)
        acc = np.concatenate([src[rows], dst[rows]]).astype(np.int64)
        nbr = np.concatenate([dst[rows], src[rows]]).astype(np.int64)
        row = np.concatenate([rows, rows]).astype(np.int64)
        order = np.argsort(acc * self.big + row, kind="stable")
        self.e_key, self.e_nbr = (acc * self.big + row)[order], nbr[order]
        mrows = np.flatnonzero(marked)
        macc = np.concatenate([src[mrows], dst[mrows]]).astype(np.int64)
        self.m_key = np.sort(macc * self.big + np.concatenate([mrows, mrows]))

    def any_marked(self, accs: np.ndarray, lo: int, i: int) -> bool:
        """Un des comptes a-t-il une transaction marquée dans les lignes [lo, i) ?"""
        k = np.searchsorted(self.m_key, accs * self.big + i) - 1
        ok = k >= 0
        prev = self.m_key[np.maximum(k, 0)]
        return bool(np.any(ok & (prev // self.big == accs) & (prev % self.big >= lo)))

    def neighbors(self, acc: int, lo: int, i: int, cap: int) -> tuple[np.ndarray, bool]:
        a = np.searchsorted(self.e_key, acc * self.big + lo)
        b = np.searchsorted(self.e_key, acc * self.big + i)
        return self.e_nbr[max(a, b - cap) : b], b - a > cap  # les plus récents d'abord retenus


def _hops(h: History, src, dst, t, query_rows, window: int, max_hops: int, nbr_cap: int, max_frontier: int):
    out = np.full(len(query_rows), -1, dtype=np.int8)  # -1 : isolé
    trunc = 0
    for q, i in enumerate(query_rows):
        lo = int(np.searchsorted(t, t[i] - window)) if window else 0
        frontier = np.unique(np.array([src[i], dst[i]], dtype=np.int64))
        seen = set(frontier.tolist())
        for hop in range(max_hops + 1):
            if h.any_marked(frontier, lo, i):
                out[q] = hop
                break
            if hop == max_hops:
                break
            nxt = []
            for acc in frontier.tolist():
                nb, cut = h.neighbors(acc, lo, i, nbr_cap)
                trunc += cut
                nxt.extend(x for x in nb.tolist() if x not in seen)
            if not nxt:
                break
            frontier = np.unique(np.array(nxt, dtype=np.int64))
            if len(frontier) > max_frontier:
                trunc += 1
                frontier = frontier[:max_frontier]
            seen.update(frontier.tolist())
    return out, trunc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="HI-Small")
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 96], help="fenêtres en heures (0 = tout le passé)")
    ap.add_argument("--max-hops", type=int, default=3)
    ap.add_argument("--legit-sample", type=int, default=20_000, help="licites tirées au hasard pour estimer le bruit")
    ap.add_argument("--nbr-cap", type=int, default=1000, help="voisins examinés au plus par compte et par saut")
    ap.add_argument("--max-frontier", type=int, default=20_000, help="comptes examinés au plus par saut")
    ap.add_argument("--max-rows", type=int, default=0, help="essai rapide : historique et validation tronqués")
    a = ap.parse_args()
    t0 = time.perf_counter()

    s = load(a.variant)
    y = load_labels(a.variant)["y"]
    X, _, _ = load_features(a.variant)
    i3 = int(np.searchsorted(s.t, 3 * 1440))
    v0, v1 = s.split_range("val")
    h0 = 0
    if a.max_rows:
        h0, v1 = max(i3, v0 - a.max_rows), min(v1, v0 + a.max_rows)
    print(f"historique [{h0}, {v0}), validation [{v0}, {v1})", flush=True)

    parts = [_alerts("strict_b", X, h0, i3)] if h0 < i3 else []
    al = np.concatenate([*parts, _alerts("strict_a", X, max(h0, i3), v0), _alerts("strict", X, v0, v1)])
    yy = np.asarray(y[h0:v1]) == 1
    src, dst = np.asarray(s.src[h0:v1]).astype(np.int64), np.asarray(s.dst[h0:v1]).astype(np.int64)
    t = np.asarray(s.t[h0:v1])
    val = np.arange(v0 - h0, v1 - h0)
    rng = np.random.default_rng(SEED)
    legit_val = val[~yy[val]]
    groups = {
        "illicites_manquees": val[yy[val] & ~al[val]],
        "illicites_detectees": val[yy[val] & al[val]],
        "licites_echantillon": np.sort(rng.choice(legit_val, size=min(a.legit_sample, len(legit_val)), replace=False)),
    }
    scale = len(legit_val) / max(len(groups["licites_echantillon"]), 1)
    labels = [str(k) for k in range(a.max_hops + 1)] + ["isole"]

    res = {}
    for hist, marked in (("alertes", al), ("etiquettes", yy)):
        h = History(src, dst, marked)
        for w in a.windows:
            key = f"{hist}_{'tout_le_passe' if w == 0 else f'{w}h'}"
            r = {}
            print(f"\n== historique : {hist}, fenêtre : {'tout le passé' if w == 0 else f'{w} h'}", flush=True)
            for g, rows in groups.items():
                hp, trunc = _hops(h, src, dst, t, rows, w * 60, a.max_hops, a.nbr_cap, a.max_frontier)
                cnt = {lab: int((hp == (k if lab != "isole" else -1)).sum()) for k, lab in enumerate(labels)}
                r[g] = {"n": len(rows), "troncatures": trunc, "sauts": cnt}
                pct = "  ".join(f"{lab} {100 * c / max(len(rows), 1):5.1f} %" for lab, c in cnt.items())
                print(f"  {g:22s} n={len(rows):7d}  {pct}  (tronqué {trunc})", flush=True)
            # si l'on alertait toutes les transactions à au plus k sauts d'un compte marqué
            cum = []
            for k in range(a.max_hops + 1):
                miss = sum(r["illicites_manquees"]["sauts"][str(j)] for j in range(k + 1))
                fp = scale * sum(r["licites_echantillon"]["sauts"][str(j)] for j in range(k + 1))
                cum.append({"max_sauts": k, "manquees_rattrapees": miss,
                            "pct_manquees": round(100 * miss / max(r["illicites_manquees"]["n"], 1), 2),
                            "licites_alertees_estimees": round(fp), "precision": round(miss / (miss + fp), 5)
                            if miss + fp else None})  # fmt: skip
                print(f"  ≤ {k} saut(s) : {miss} manquées rattrapées ({cum[-1]['pct_manquees']} %), "
                      f"≈ {round(fp)} licites alertées en plus, précision {cum[-1]['precision']}")  # fmt: skip
            r["cumul"] = cum
            res[key] = r

    payload = {
        "run": run_info(SEED),
        "variant": a.variant,
        "args": vars(a),
        "n_legit_val": len(legit_val),
        "history_rows": [h0, v0],
        "val_rows": [v0, v1],
        "seconds": round(time.perf_counter() - t0, 1),
        "results": res,
    }
    out = ROOT / "out" / "diag"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"diag_propagation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=lambda o: o.item()) + "\n")
    print(f"\n{path}")


if __name__ == "__main__":
    main()
