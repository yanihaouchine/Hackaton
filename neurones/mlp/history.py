"""Historique brut des comptes pour le modèle séquentiel (étape 3) : causal, tous formats de paiement.

Pour chaque transaction j, on retient les K dernières transactions de l'émetteur et du destinataire,
**strictement antérieures** dans le flux trié (même convention que le moteur de arbre/ : les lignes déjà
traitées). On ne stocke que leurs indices (int32) ; les caractéristiques de chaque transaction passée sont
calculées à la volée, par lot, avec des échelles fixes (aucune statistique apprise sur les données).

Construction vectorisée : une ligne d'événement par (compte, transaction) (une seule pour un auto-virement),
tri stable par (compte, transaction), puis décalages de 1 à K à l'intérieur de chaque compte.
"""

from __future__ import annotations

import numpy as np

K = 8
N_FMT = 7
# sens (envoi), log montant, log délai, format (one-hot), même contrepartie, auto-virement, changement de devise
EVENT_DIM = 3 + N_FMT + 3


def build_index(src: np.ndarray, dst: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """(hist_src, hist_dst) : (n, k) indices des k transactions précédentes du compte (-1 si absente),
    de la plus récente à la plus ancienne."""
    n = len(src)
    idx = np.arange(n, dtype=np.int64)
    self_tx = src == dst
    # Événements : côté émetteur pour toutes, côté destinataire sauf auto-virement (même compte, même ligne)
    acc = np.concatenate([src, dst[~self_tx]]).astype(np.int64)
    tx = np.concatenate([idx, idx[~self_tx]])
    side = np.concatenate([np.zeros(n, np.int8), np.ones(int((~self_tx).sum()), np.int8)])
    order = np.lexsort((side, tx, acc))  # par compte, puis transaction
    acc, tx, side = acc[order], tx[order], side[order]
    hist_src = np.full((n, k), -1, dtype=np.int32)
    hist_dst = np.full((n, k), -1, dtype=np.int32)
    is_src = side == 0
    is_dst = (side == 1) | (side == 0) & self_tx[tx]  # auto-virement : l'événement émetteur sert aux deux côtés
    for s in range(1, k + 1):
        prev = np.full(len(acc), -1, dtype=np.int64)
        prev[s:] = np.where(acc[s:] == acc[:-s], tx[:-s], -1)
        hist_src[tx[is_src], s - 1] = prev[is_src]
        hist_dst[tx[is_dst], s - 1] = prev[is_dst]
    # Un décalage peut retomber sur la même transaction (rien d'antérieur) : jamais, car tx est strictement
    # croissant au sein d'un compte une fois les auto-virements dédoublonnés ; on le vérifie quand même.
    for h in (hist_src, hist_dst):
        rows = np.arange(n)[:, None]
        assert np.all((h < 0) | (h < rows)), "historique non causal"
    return hist_src, hist_dst


def event_features(hist: np.ndarray, rows: np.ndarray, account: np.ndarray, other: np.ndarray, stream) -> np.ndarray:
    """(len(rows), K, EVENT_DIM) : caractéristiques des transactions passées du compte `account[rows]`,
    vues depuis la transaction courante (délai, contrepartie commune avec `other[rows]`)."""
    h = hist[rows]  # (b, K)
    valid = h >= 0
    hh = np.where(valid, h, 0)
    t, src, dst = np.asarray(stream.t), np.asarray(stream.src), np.asarray(stream.dst)
    acc = account[rows][:, None]
    out_dir = src[hh] == acc
    cp = np.where(out_dir, dst[hh], src[hh])
    f = np.zeros((*h.shape, EVENT_DIM), dtype=np.float32)
    f[..., 0] = out_dir
    f[..., 1] = np.log1p(np.asarray(stream.usd_paid)[hh]) / 10.0
    f[..., 2] = np.log1p(np.maximum(t[rows][:, None] - t[hh], 0)) / 10.0
    fmt = np.asarray(stream.fmt)[hh]
    f[..., 3 : 3 + N_FMT] = fmt[..., None] == np.arange(N_FMT)
    f[..., 3 + N_FMT] = cp == other[rows][:, None]
    f[..., 4 + N_FMT] = src[hh] == dst[hh]
    f[..., 5 + N_FMT] = np.asarray(stream.cur_paid)[hh] != np.asarray(stream.cur_recv)[hh]
    f[~valid] = 0.0
    return f


def batch(hist_src, hist_dst, rows, stream) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Séquences émetteur et destinataire d'un lot, avec leurs masques de validité."""
    src, dst = np.asarray(stream.src), np.asarray(stream.dst)
    es = event_features(hist_src, rows, src, dst, stream)
    ed = event_features(hist_dst, rows, dst, src, stream)
    return es, hist_src[rows] >= 0, ed, hist_dst[rows] >= 0
