"""Frugalité du MLP livré (ONNX) et de la forêt de arbre/, mesurés sur cette machine avec le même protocole.

Protocole repris de la v1 de neurones/ (scripts/bench.py, commit 0c12204) : 1 thread, 3 000 appels à
1 transaction (latence p50/p99), 200 000 transactions de validation en lots de 4 096 (débit).
Coût du modèle seul : le moteur de features C, commun aux deux, s'y ajoute.
Usage (depuis neurones/) : python -m mlp.bench [--model h64-32_n100_s42]  ->  results/bench_<machine>.json
"""

from __future__ import annotations

import argparse
import ctypes
import platform
import resource
import sys
import time

import numpy as np
import onnxruntime as ort

from mlp.common import ARTIFACTS, load_features, load_stream, write_result

N_SINGLE = 3000
N_BATCH = 200_000
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


def max_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024)  # octets sur macOS, Ko sur Linux


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    a = p.parse_args()

    from aml.engine import load_lib
    from aml.forest import c_raw
    from aml.model import load_bundle, load_cascade

    X, _, cols = load_features()
    stream, _ = load_stream()
    lo, _ = stream.split_range("val")
    x_all = np.ascontiguousarray(X[lo : lo + N_BATCH], dtype=np.float32)  # 104 colonnes (forêt)
    x_mlp = np.ascontiguousarray(x_all[:, cols])  # 96 colonnes (MLP)

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    onnx_path = ARTIFACTS / a.model / "model.onnx"
    t = time.perf_counter()
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t) * 1e3
    mlp = timed(lambda b: sess.run(None, {"features": b}), x_mlp)
    rss_mlp = max_rss_mb()

    bundle = load_bundle("strict")
    lib, st = load_lib(), bundle.forest.to_c()

    def forest_raw(b: np.ndarray) -> np.ndarray:
        out = np.empty(len(b), dtype=np.float64)
        lib.aml_forest_raw(ctypes.byref(st), b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(b), b.shape[1],
                           out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))  # fmt: skip
        return out

    assert np.allclose(forest_raw(x_all[:1000]), c_raw(bundle.forest, x_all[:1000]))
    forest = timed(forest_raw, x_all)

    pre_forest, cut = load_cascade(bundle)
    st_pre = pre_forest.to_c()

    def cascade_raw(b: np.ndarray) -> np.ndarray:
        r0 = np.empty(len(b), dtype=np.float64)
        lib.aml_forest_raw(ctypes.byref(st_pre), b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(b),
                           b.shape[1], r0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))  # fmt: skip
        keep = np.flatnonzero(r0 >= cut)
        if len(keep):
            r0[keep] = forest_raw(np.ascontiguousarray(b[keep]))
        return r0

    passed = float(np.mean(c_raw(pre_forest, x_all) >= cut))
    cascade = {**timed(cascade_raw, x_all), "share_sent_to_full_forest": passed}

    payload = {
        "threads": 1,
        "energy": "non mesuré (powermetrics demande les droits administrateur sur macOS)",
        "mlp_onnx": {
            "model": a.model,
            "file_bytes": onnx_path.stat().st_size,
            "load_ms": round(load_ms, 1),
            **mlp,
            "max_rss_mb_process": rss_mlp,
        },  # fmt: skip
        "forest_strict_arbre": {
            "file_bytes": (bundle.path / "forest.amlz").stat().st_size,
            "n_trees": bundle.info.get("n_trees"),
            "evaluator": "C (aml_forest_raw, structure préparée une fois), sans cascade",
            **forest,
        },  # fmt: skip
        "forest_strict_cascade_arbre": {
            "file_bytes": (bundle.path / "forest.amlz").stat().st_size + (bundle.path / "cascade.amlz").stat().st_size,
            "evaluator": "mini-forêt puis forêt complète au-delà du seuil (configuration temps réel livrée)",
            **cascade,
        },
        "note": "Coût du modèle seul ; le moteur de features C, commun aux deux, s'y ajoute.",
    }
    path = write_result(f"bench_{platform.machine()}", payload, seed=42)
    for k in ("mlp_onnx", "forest_strict_arbre", "forest_strict_cascade_arbre"):
        d = payload[k]
        print(f"{k:28s} {d['file_bytes'] / 1024:8.0f} Ko  p50 {d['latency_us_p50']:7.2f} µs  "
              f"p99 {d['latency_us_p99']:7.2f} µs  {d['throughput_tx_s']:>10,} tx/s")  # fmt: skip
    print(path)


if __name__ == "__main__":
    main()
