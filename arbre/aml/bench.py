"""Frugalité mesurée : débit, latence par transaction (p50/p99/p99,9), RAM, tailles, énergie si mesurable.

Tout est mesuré sur la machine courante avec un seul thread (le moteur est mono-thread) ; les résultats
vont dans results/bench_<machine>.json. Énergie : RAPL si lisible, sinon puissance saisie au wattmètre
(--watts-idle / --watts-load), sinon « non mesuré ».
"""

from __future__ import annotations

import os
import platform
import resource
import time
from pathlib import Path

import numpy as np

from aml.config import DATA, EngineConfig
from aml.data import load
from aml.engine import TX_FIELDS, Engine, lib_path
from aml.model import load_bundle, load_cascade
from aml.provenance import write_result

RAPL = Path("/sys/class/powercap/intel-rapl:0/energy_uj")


def _rapl() -> float | None:
    try:
        return int(RAPL.read_text()) / 1e6
    except (OSError, ValueError):
        return None


def _proc_status() -> dict[str, int]:
    out = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile"):
                out[k] = int(v.split()[0]) // 1024  # MiB
    except OSError:
        pass
    return out


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def run(variant: str, model: str = "strict", chunk: int = 1 << 16, watts_idle: float | None = None,
        watts_load: float | None = None, latency_sample: int = 200_000, tag: str = "",
        cascade: bool = True) -> dict:  # fmt: skip
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    s = load(variant)
    b = load_bundle(model)
    cfg = EngineConfig(**{k: v for k, v in b.info["engine_config"].items()})
    eng = Engine.for_stream(s, cfg)
    eng.set_forest(b.forest)
    casc = load_cascade(b) if cascade else None
    if casc is not None:
        eng.set_cascade(*casc)
    feats = np.empty((chunk, eng.n_features), dtype=np.float32)
    scores = np.empty(chunk, dtype=np.float64)
    timing = np.empty(chunk, dtype=np.int64)
    lat_lo = s.split_range("test")[0]  # latences mesurées sur la période de test (état déjà chaud)
    lat = []
    alerts = 0
    e0, w0, c0 = _rapl(), time.perf_counter(), time.process_time()
    for a in range(0, s.n, chunk):
        bb = min(s.n, a + chunk)
        n = bb - a
        cols = {k: getattr(s, k)[a:bb] for k, _ in TX_FIELDS}
        want_t = a + n > lat_lo and sum(len(x) for x in lat) < latency_sample
        eng.process(cols, feats[:n], scores=scores[:n], timing=timing[:n] if want_t else None)
        if want_t:
            lat.append(timing[max(0, lat_lo - a) : n].copy())
        alerts += int((scores[:n] >= b.threshold).sum())
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    e1 = _rapl()
    ns = np.concatenate(lat)[:latency_sample] if lat else np.array([0])
    status = _proc_status()

    # latence de bout en bout côté Python (appel ctypes compris), lot de 1 et lot de 1 024
    eng2 = Engine.for_stream(s, cfg)
    eng2.set_forest(b.forest)
    if casc is not None:
        eng2.set_cascade(*casc)
    warm = s.split_range("test")[0]
    snap = sorted((DATA / "state" / variant).glob(f"*_{warm}.snap"))
    try:  # état chaud : instantané de la démo s'il existe (même moteur), sinon calcul
        eng2.restore(snap[0])
    except (IndexError, OSError):
        eng2.process_stream(s, 0, warm, chunk=1 << 20)
    one_x, one_s = np.empty((1, eng2.n_features), np.float32), np.empty(1, np.float64)
    per_call = []
    for i in range(warm, warm + 20_000):
        cols = {k: getattr(s, k)[i : i + 1] for k, _ in TX_FIELDS}
        t = time.perf_counter_ns()
        eng2.process(cols, one_x, scores=one_s)
        per_call.append(time.perf_counter_ns() - t)
    per_call = np.array(per_call)
    blk_x, blk_s = np.empty((1024, eng2.n_features), np.float32), np.empty(1024, np.float64)
    per_blk = []
    for a in range(warm + 20_000, warm + 20_000 + 1024 * 100, 1024):
        cols = {k: getattr(s, k)[a : a + 1024] for k, _ in TX_FIELDS}
        t = time.perf_counter_ns()
        eng2.process(cols, blk_x, scores=blk_s)
        per_blk.append(time.perf_counter_ns() - t)
    per_blk = np.array(per_blk) / 1024

    energy = {"methode": "non mesuré"}
    if e0 is not None and e1 is not None and e1 > e0:
        j = e1 - e0
        energy = {"methode": "RAPL (paquet CPU)", "joules": j, "uj_per_tx": 1e6 * j / s.n}
    elif watts_idle is not None and watts_load is not None:
        j = (watts_load - watts_idle) * wall
        energy = {"methode": "wattmètre (charge − repos) × durée", "watts_idle": watts_idle,
                  "watts_load": watts_load, "joules": j, "uj_per_tx": 1e6 * j / s.n}  # fmt: skip
    res = {
        "variant": variant,
        "model": model,
        "n_tx": s.n,
        "threads": 1,
        "cascade": casc is not None,
        "stream": {"wall_s": wall, "cpu_s": cpu, "tx_per_s": s.n / wall, "alerts": alerts, "chunk": chunk},  # fmt: skip
        "latency_engine_ns": {
            "p50": _pct(ns, 50),
            "p99": _pct(ns, 99),
            "p999": _pct(ns, 99.9),
            "max": float(ns.max()),
            "n": int(len(ns)),
        },  # fmt: skip
        "latency_python_call_us": {
            "batch1_p50": _pct(per_call, 50) / 1e3,
            "batch1_p99": _pct(per_call, 99) / 1e3,
            "batch1024_per_tx_p50": _pct(per_blk, 50) / 1e3,
        },  # fmt: skip
        "memory_mib": {
            "engine_state": eng.state_bytes() / 2**20,
            **status,
            "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        },  # fmt: skip
        "sizes_bytes": {
            "forest": (b.path / "forest.amlz").stat().st_size,
            "engine_so": lib_path().stat().st_size,
            "lgbm_txt": (b.path / "lgbm.txt").stat().st_size if (b.path / "lgbm.txt").exists() else None,
        },  # fmt: skip
        "energy": energy,
        "engine_stats": eng.stats(),
        "platform": platform.platform(),
    }
    res["tag"] = tag
    write_result(f"bench_{platform.node()}_{platform.machine()}{'_' + tag if tag else ''}", res)
    eng.close()
    eng2.close()
    return res
