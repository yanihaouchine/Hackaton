"""Frugalité du MLP livré (ONNX), mesurée sur cette machine, à côté de la forêt de arbre/ mesurée de la même façon.
Les deux modèles reçoivent les mêmes features (sorties du moteur C) : on ne mesure ici que le coût du modèle.
Usage (depuis neurones/) : ../.venv/bin/python scripts/bench.py
Sortie : results/bench_mlp_<machine>.json
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nn.data import RESULTS, load

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_SINGLE = 3000  # appels à 1 transaction (latence)
N_BATCH = 200_000  # transactions en lots (débit)
BATCH = 4096


def pct(xs: list[float], q: float) -> float:
    return round(float(np.percentile(xs, q)), 2)


def timed(fn, x: np.ndarray) -> dict:
    """Latence par appel à 1 transaction (p50, p99, en µs) et débit en lots (tx/s)."""
    for i in range(50):  # chauffe
        fn(x[i : i + 1])
    lat = []
    for i in range(N_SINGLE):
        t = time.perf_counter()
        fn(x[i : i + 1])
        lat.append((time.perf_counter() - t) * 1e6)
    t = time.perf_counter()
    for i in range(0, N_BATCH, BATCH):
        fn(x[i : i + BATCH])
    secs = time.perf_counter() - t
    return {"latency_us_p50": pct(lat, 50), "latency_us_p99": pct(lat, 99), "throughput_tx_s": round(N_BATCH / secs)}


def main() -> None:
    import onnxruntime as ort
    from aml.forest import c_raw
    from aml.model import load_bundle
    from aml.provenance import run_info

    ds = load()
    lo, _ = ds.ranges["val"]
    x_all = np.ascontiguousarray(ds.X[lo : lo + N_BATCH], dtype=np.float32)  # 104 colonnes (forêt)
    x_mlp = np.ascontiguousarray(x_all[:, ds.cols])  # 96 colonnes (MLP)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    t = time.perf_counter()
    sess = ort.InferenceSession(str(ARTIFACTS / "mlp.onnx"), so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t) * 1e3
    mlp = timed(lambda b: sess.run(None, {"features": b}), x_mlp)
    rss_mlp = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    bundle = load_bundle("strict")
    forest = timed(lambda b: c_raw(bundle.forest, b), x_all)

    out = {
        "run": run_info(),
        "machine": platform.machine(),
        "threads": 1,
        "energy": "non mesuré (RAPL illisible sans droits root sur cette machine)",
        "mlp_onnx": {
            "file_bytes": (ARTIFACTS / "mlp.onnx").stat().st_size,
            "load_ms": round(load_ms, 1),
            **mlp,
            "max_rss_mb_process": round(rss_mlp),
        },
        "forest_strict_arbre": {
            "file_bytes": (bundle.path / "forest.amlz").stat().st_size,
            "n_trees": bundle.info.get("n_trees"),
            "evaluator": "C (aml_forest_raw), sans cascade",
            **forest,
        },
        "note": "Coût du modèle seul ; le moteur de features C, commun aux deux, s'y ajoute.",
    }
    RESULTS.mkdir(exist_ok=True)
    name = f"bench_mlp_{platform.machine()}.json"
    (RESULTS / name).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
