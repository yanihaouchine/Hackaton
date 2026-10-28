import numpy as np

from mlp.history import build_index


def brute_force(src: np.ndarray, dst: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Référence naïve : pour chaque ligne, on remonte le passé et on garde les k dernières du compte."""
    n = len(src)
    hs, hd = np.full((n, k), -1), np.full((n, k), -1)
    for j in range(n):
        for h, a in ((hs, src[j]), (hd, dst[j])):
            past = [i for i in range(j - 1, -1, -1) if src[i] == a or dst[i] == a][:k]
            h[j, : len(past)] = past
    return hs, hd


def toy(n: int = 400, n_acc: int = 12, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_acc, n)
    dst = np.where(rng.random(n) < 0.15, src, rng.integers(0, n_acc, n))  # 15 % d'auto-virements
    return src.astype(np.int32), dst.astype(np.int32)


def test_matches_brute_force() -> None:
    src, dst = toy()
    hs, hd = build_index(src, dst, k=5)
    rs, rd = brute_force(src, dst, k=5)
    assert np.array_equal(hs, rs) and np.array_equal(hd, rd)


def test_strictly_past_only() -> None:
    src, dst = toy()
    hs, hd = build_index(src, dst, k=8)
    rows = np.arange(len(src))[:, None]
    assert np.all(hs < rows) and np.all(hd < rows)  # -1 (absent) est aussi < rows


def test_future_does_not_change_past() -> None:
    """Couper le flux en m ne change pas l'historique des m premières lignes."""
    src, dst = toy()
    full_s, full_d = build_index(src, dst, k=8)
    for m in (1, 50, 233):
        part_s, part_d = build_index(src[:m], dst[:m], k=8)
        assert np.array_equal(part_s, full_s[:m]) and np.array_equal(part_d, full_d[:m])
