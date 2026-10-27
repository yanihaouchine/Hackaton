"""Baseline B0 : règles expertes lisibles, seuils fixés sur la validation (montée de coordonnées).

Chaque règle porte sur une feature du moteur et ne s'applique qu'aux formats du graphe de motifs.
Ces règles servent aussi d'indicateurs nommés dans les rapports.
"""

from __future__ import annotations

import numpy as np

from aml.evaluate import binary_at

# nom → (feature, sens, seuils candidats, libellé français avec {v})
RULES = {
    "eventail_sortant": ("g_s_outdeg_l", ">=", [2, 3, 4, 5, 6, 8, 10, 15],
                         "le compte émetteur a payé au moins {v} bénéficiaires distincts en 96 h"),
    "eventail_entrant": ("g_r_indeg_l", ">=", [2, 3, 4, 5, 6, 8, 10, 15],
                         "le compte destinataire a reçu d'au moins {v} émetteurs distincts en 96 h"),
    "passage_de_fonds": ("f_best_err", "<=", [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                         "le montant reproduit à {v:.1%} près une somme reçue dans les 96 h (compte de passage)"),
    "chaine_de_passage": ("f_chain_depth", ">=", [1, 2, 3, 4],
                          "chaîne d'au moins {v} transferts successifs de montants quasi identiques"),
    "cycle": ("m_cyc_n", ">=", [1, 2],
              "le virement referme un circuit de fonds revenant au compte d'origine"),
    "dispersion_collecte": ("m_sg", ">=", [1, 2, 3, 4],
                            "au moins {v} intermédiaires relient une même source à ce destinataire"),
    "collecte_dispersion": ("g_s_indeg_l", ">=", [2, 3, 4, 5, 8],
                            "le compte émetteur redistribue des fonds reçus d'au moins {v} émetteurs"),
}  # fmt: skip


def rule_masks(X, names: list[str], thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    col = {n: i for i, n in enumerate(names)}
    in_graph = np.asarray(X[:, col["tx_is_ach"]]) > 0
    out = {}
    for r, v in thresholds.items():
        feat, op, _, _ = RULES[r]
        x = np.asarray(X[:, col[feat]])
        out[r] = in_graph & ((x >= v) if op == ">=" else (x <= v))
    return out


def fit_rules(X, names, y, passes: int = 3) -> tuple[dict[str, float], dict]:
    """Montée de coordonnées sur le F1 : chaque règle est désactivée ou reçoit son meilleur seuil."""
    y = np.asarray(y)
    col = {n: i for i, n in enumerate(names)}
    in_graph = np.asarray(X[:, col["tx_is_ach"]]) > 0
    cand = {}
    for r, (feat, op, grid, _) in RULES.items():
        x = np.asarray(X[:, col[feat]])
        cand[r] = {v: in_graph & ((x >= v) if op == ">=" else (x <= v)) for v in grid}
    chosen: dict[str, float] = {}

    def score(sel):
        m = np.zeros(len(y), dtype=bool)
        for r, v in sel.items():
            m |= cand[r][v]
        return binary_at(y, m.astype(np.float32), 0.5)["f1"]

    best = 0.0
    for _ in range(passes):
        for r in RULES:
            options = [None, *cand[r]]
            res = []
            for v in options:
                sel = {k: w for k, w in chosen.items() if k != r}
                if v is not None:
                    sel[r] = v
                res.append((score(sel), v))
            f, v = max(res, key=lambda z: z[0])
            if f > best or (v is None and r in chosen and f >= best):
                best = f
                chosen = {k: w for k, w in chosen.items() if k != r}
                if v is not None:
                    chosen[r] = v
    m = np.zeros(len(y), dtype=bool)
    for r, v in chosen.items():
        m |= cand[r][v]
    return chosen, binary_at(y, m.astype(np.float32), 0.5)
