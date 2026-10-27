"""Chargement du moteur C (ctypes) : compilé localement sans -march, une bibliothèque par architecture."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from ctypes import POINTER, c_double, c_float, c_int16, c_int32, c_int64, c_uint8
from pathlib import Path

import numpy as np

from aml.config import BUILD, FORMATS, EngineConfig

SRC = Path(__file__).with_name("aml_engine.c")
HDR = Path(__file__).with_name("aml_engine.h")
CFLAGS = ["-O2", "-std=c99", "-ffp-contract=off", "-fPIC", "-shared", "-s"]  # jamais -march ; pas de FMA implicite


class _Config(ctypes.Structure):
    _fields_ = [
        ("n_accounts", c_int32), ("win_short", c_int32), ("win_mid", c_int32), ("win_long", c_int32),
        ("motif_fmt_mask", c_int32), ("ach_code", c_int32), ("ring_bits", c_int32), ("pair_bits", c_int32),
        ("pair_ttl", c_int32), ("scan_cap", c_int32), ("fan_cap", c_int32), ("cycle_max_len", c_int32),
        ("cycle_budget", c_int32), ("chain_tol", c_float),
    ]  # fmt: skip


class _Batch(ctypes.Structure):
    _fields_ = [
        ("n", c_int64), ("t", POINTER(c_int32)), ("src", POINTER(c_int32)), ("dst", POINTER(c_int32)),
        ("usd_paid", POINTER(c_float)), ("usd_recv", POINTER(c_float)), ("amt_paid", POINTER(c_double)),
        ("cur_paid", POINTER(c_uint8)), ("cur_recv", POINTER(c_uint8)), ("fmt", POINTER(c_uint8)),
    ]  # fmt: skip


class _Kyc(ctypes.Structure):
    _fields_ = [
        ("bank", POINTER(c_int32)), ("entity", POINTER(c_int32)), ("etype", POINTER(c_uint8)),
        ("nacc", POINTER(c_int32)), ("country", POINTER(c_int16)),
    ]  # fmt: skip


class _Forest(ctypes.Structure):
    _fields_ = [
        ("n_trees", c_int32), ("n_nodes", c_int32), ("root", POINTER(c_int32)), ("feat", POINTER(c_int32)),
        ("thr", POINTER(c_double)), ("left", POINTER(c_int32)), ("right", POINTER(c_int32)),
        ("default_left", POINTER(c_uint8)), ("value", POINTER(c_double)),
    ]  # fmt: skip


def aml_forest_struct(n_trees: int, n_nodes: int, a: dict[str, np.ndarray]) -> _Forest:
    return _Forest(
        n_trees=n_trees, n_nodes=n_nodes, root=_ptr(a["root"], c_int32), feat=_ptr(a["feat"], c_int32),
        thr=_ptr(a["thr"], c_double), left=_ptr(a["left"], c_int32), right=_ptr(a["right"], c_int32),
        default_left=_ptr(a["default_left"], c_uint8), value=_ptr(a["value"], c_double),
    )  # fmt: skip


def lib_path() -> Path:
    return BUILD / f"libamlengine-{platform.machine()}.so"


def build(force: bool = False) -> Path:
    """Compile la bibliothèque si elle manque ou si les sources sont plus récentes."""
    out = lib_path()
    newest = max(SRC.stat().st_mtime, HDR.stat().st_mtime)
    if out.exists() and not force and out.stat().st_mtime >= newest:
        return out
    cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        raise RuntimeError("aucun compilateur C trouvé (cc, gcc ou clang) : lancer scripts/build_engine.sh")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    subprocess.run([cc, *CFLAGS, "-o", str(tmp), str(SRC), "-lm"], check=True)
    tmp.replace(out)
    return out


_LIB = None


def load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build()))
        lib.aml_new.restype = ctypes.c_void_p
        lib.aml_new.argtypes = [POINTER(_Config), POINTER(_Kyc)]
        lib.aml_free.argtypes = [ctypes.c_void_p]
        lib.aml_n_features.restype = c_int32
        lib.aml_feature_name.restype = ctypes.c_char_p
        lib.aml_feature_name.argtypes = [c_int32]
        lib.aml_process.restype = ctypes.c_int
        lib.aml_process.argtypes = [ctypes.c_void_p, POINTER(_Batch), POINTER(c_float)]
        lib.aml_state_bytes.restype = c_int64
        lib.aml_state_bytes.argtypes = [ctypes.c_void_p]
        lib.aml_n_stats.restype = c_int32
        lib.aml_stat_name.restype = ctypes.c_char_p
        lib.aml_stat_name.argtypes = [c_int32]
        lib.aml_stats.argtypes = [ctypes.c_void_p, POINTER(c_int64)]
        lib.aml_process_scored.restype = ctypes.c_int
        lib.aml_process_scored.argtypes = [ctypes.c_void_p, POINTER(_Batch), POINTER(c_float), POINTER(c_double)]
        lib.aml_set_forest.restype = ctypes.c_int
        lib.aml_set_forest.argtypes = [ctypes.c_void_p, POINTER(_Forest)]
        lib.aml_forest_raw.restype = None
        lib.aml_forest_raw.argtypes = [POINTER(_Forest), POINTER(c_float), c_int64, c_int32, POINTER(c_double)]
        lib.aml_set_timing.restype = None
        lib.aml_set_timing.argtypes = [ctypes.c_void_p, POINTER(c_int64)]
        lib.aml_snapshot.restype = c_int64
        lib.aml_snapshot.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.aml_restore.restype = ctypes.c_int
        lib.aml_restore.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.aml_set_cascade.restype = ctypes.c_int
        lib.aml_set_cascade.argtypes = [ctypes.c_void_p, POINTER(_Forest), c_double]
        lib.aml_set_seq_out.restype = None
        lib.aml_set_seq_out.argtypes = [ctypes.c_void_p, POINTER(c_int64)]
        lib.aml_n_hindsight.restype = c_int32
        lib.aml_hindsight_name.restype = ctypes.c_char_p
        lib.aml_hindsight_name.argtypes = [c_int32]
        lib.aml_hindsight.restype = ctypes.c_int
        lib.aml_hindsight.argtypes = [ctypes.c_void_p, POINTER(_Batch), POINTER(c_int64), POINTER(c_float)]
        _LIB = lib
    return _LIB


def feature_names() -> list[str]:
    lib = load_lib()
    return [lib.aml_feature_name(i).decode() for i in range(lib.aml_n_features())]


def hindsight_names() -> list[str]:
    lib = load_lib()
    return [lib.aml_hindsight_name(i).decode() for i in range(lib.aml_n_hindsight())]


def _ptr(a: np.ndarray, ctype):
    return a.ctypes.data_as(POINTER(ctype))


TX_FIELDS = (
    ("t", np.int32), ("src", np.int32), ("dst", np.int32), ("usd_paid", np.float32), ("usd_recv", np.float32),
    ("amt_paid", np.float64), ("cur_paid", np.uint8), ("cur_recv", np.uint8), ("fmt", np.uint8),
)  # fmt: skip


class Engine:
    """État de flux borné ; process() calcule les features puis met l'état à jour, transaction par transaction."""

    def __init__(self, n_accounts: int, kyc: dict[str, np.ndarray], cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.lib = load_lib()
        c = self.cfg
        conf = _Config(
            n_accounts=n_accounts, win_short=c.win_short, win_mid=c.win_mid, win_long=c.win_long,
            motif_fmt_mask=c.fmt_mask(), ach_code=FORMATS.index("ACH"), ring_bits=c.ring_bits,
            pair_bits=c.pair_bits, pair_ttl=c.pair_ttl, scan_cap=c.scan_cap, fan_cap=c.fan_cap,
            cycle_max_len=c.cycle_max_len, cycle_budget=c.cycle_budget, chain_tol=c.chain_tol,
        )  # fmt: skip
        ky = {
            "bank": np.ascontiguousarray(kyc["acc_bank"], dtype=np.int32),
            "entity": np.ascontiguousarray(kyc["acc_entity"], dtype=np.int32),
            "etype": np.ascontiguousarray(kyc["acc_etype"], dtype=np.uint8),
            "nacc": np.ascontiguousarray(kyc["acc_nacc"], dtype=np.int32),
            "country": np.ascontiguousarray(kyc["acc_country"], dtype=np.int16),
        }
        k = _Kyc(
            bank=_ptr(ky["bank"], c_int32), entity=_ptr(ky["entity"], c_int32),
            etype=_ptr(ky["etype"], c_uint8), nacc=_ptr(ky["nacc"], c_int32),
            country=_ptr(ky["country"], c_int16),
        )  # fmt: skip
        self.h = self.lib.aml_new(ctypes.byref(conf), ctypes.byref(k))
        if not self.h:
            raise MemoryError("aml_new a échoué (configuration invalide ou mémoire insuffisante)")
        self.n_features = self.lib.aml_n_features()
        self.names = feature_names()

    @classmethod
    def for_stream(cls, stream, cfg: EngineConfig | None = None) -> Engine:
        kyc = {k: getattr(stream, k) for k in ("acc_bank", "acc_entity", "acc_etype", "acc_nacc", "acc_country")}
        return cls(len(stream.acc_etype), kyc, cfg)

    def set_forest(self, forest) -> None:
        k = len(forest.engine_features)
        if forest.engine_features != self.names[:k] or int(np.max(forest.feat)) >= k:
            raise ValueError("forêt entraînée sur un autre jeu de features que ce moteur")
        self._forest_struct = forest.to_c()
        rc = self.lib.aml_set_forest(self.h, ctypes.byref(self._forest_struct))
        if rc != 0:
            raise RuntimeError(f"aml_set_forest a échoué (code {rc})")
        self.has_forest = True

    def _batch(self, cols: dict[str, np.ndarray]):
        arrs = {k: np.ascontiguousarray(cols[k], dtype=dt) for k, dt in TX_FIELDS}
        b = _Batch(
            n=len(arrs["t"]), t=_ptr(arrs["t"], c_int32), src=_ptr(arrs["src"], c_int32),
            dst=_ptr(arrs["dst"], c_int32), usd_paid=_ptr(arrs["usd_paid"], c_float),
            usd_recv=_ptr(arrs["usd_recv"], c_float), amt_paid=_ptr(arrs["amt_paid"], c_double),
            cur_paid=_ptr(arrs["cur_paid"], c_uint8), cur_recv=_ptr(arrs["cur_recv"], c_uint8),
            fmt=_ptr(arrs["fmt"], c_uint8),
        )  # fmt: skip
        return b, arrs

    def hindsight(self, cols: dict[str, np.ndarray], seq_at: np.ndarray) -> np.ndarray:
        """Revue J+1 : features rétrospectives d'un lot déjà traité, avec l'état courant (décision)."""
        b, _keep = self._batch(cols)
        seq_at = np.ascontiguousarray(seq_at, dtype=np.int64)
        out = np.empty((b.n, self.lib.aml_n_hindsight()), dtype=np.float32)
        rc = self.lib.aml_hindsight(self.h, ctypes.byref(b), _ptr(seq_at, c_int64), _ptr(out, c_float))
        if rc != 0:
            raise RuntimeError(f"aml_hindsight a échoué (code {rc})")
        return out

    def set_cascade(self, pre, raw_cut: float) -> None:
        """Mini-modèle de pré-filtrage (cascade) ; pre=None la désactive."""
        if pre is None:
            self.lib.aml_set_cascade(self.h, None, 0.0)
            return
        self._pre_struct = pre.to_c()
        rc = self.lib.aml_set_cascade(self.h, ctypes.byref(self._pre_struct), float(raw_cut))
        if rc != 0:
            raise RuntimeError(f"aml_set_cascade a échoué (code {rc})")

    def process(self, cols: dict[str, np.ndarray], out: np.ndarray | None = None,
                scores: np.ndarray | None = None, timing: np.ndarray | None = None,
                seq_at: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
        """Features (et scores si `scores` est fourni et une forêt attachée) ; `timing` reçoit les ns/tx ;
        `seq_at` reçoit le repère de séquence de chaque transaction (revue J+1)."""
        arrs = {k: np.ascontiguousarray(cols[k], dtype=dt) for k, dt in TX_FIELDS}
        n = len(arrs["t"])
        if out is None:
            out = np.empty((n, self.n_features), dtype=np.float32)
        if out.shape != (n, self.n_features) or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise ValueError("tableau de sortie incompatible")
        sp = None
        if scores is not None:
            if scores.shape != (n,) or scores.dtype != np.float64:
                raise ValueError("tableau de scores incompatible")
            sp = _ptr(scores, c_double)
        if timing is not None:
            if timing.shape != (n,) or timing.dtype != np.int64:
                raise ValueError("tableau de latences incompatible")
            self.lib.aml_set_timing(self.h, _ptr(timing, c_int64))
        if seq_at is not None:
            if seq_at.shape != (n,) or seq_at.dtype != np.int64:
                raise ValueError("tableau seq_at incompatible")
            self.lib.aml_set_seq_out(self.h, _ptr(seq_at, c_int64))
        b = _Batch(
            n=n, t=_ptr(arrs["t"], c_int32), src=_ptr(arrs["src"], c_int32), dst=_ptr(arrs["dst"], c_int32),
            usd_paid=_ptr(arrs["usd_paid"], c_float), usd_recv=_ptr(arrs["usd_recv"], c_float),
            amt_paid=_ptr(arrs["amt_paid"], c_double), cur_paid=_ptr(arrs["cur_paid"], c_uint8),
            cur_recv=_ptr(arrs["cur_recv"], c_uint8), fmt=_ptr(arrs["fmt"], c_uint8),
        )  # fmt: skip
        rc = self.lib.aml_process_scored(self.h, ctypes.byref(b), _ptr(out, c_float), sp)
        if timing is not None:
            self.lib.aml_set_timing(self.h, None)
        if seq_at is not None:
            self.lib.aml_set_seq_out(self.h, None)
        if rc != 0:
            raise RuntimeError(f"aml_process a échoué (code {rc})")
        return out

    def process_stream(self, stream, lo: int = 0, hi: int | None = None, out: np.ndarray | None = None,
                       chunk: int = 1 << 18, progress: bool = False,
                       scores: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
        """Traite stream[lo:hi] par blocs ; `out` peut être un memmap (RAM bornée)."""
        hi = stream.n if hi is None else hi
        if out is None:
            out = np.empty((hi - lo, self.n_features), dtype=np.float32)
        t0 = time.perf_counter()
        for a in range(lo, hi, chunk):
            b = min(hi, a + chunk)
            cols = {k: getattr(stream, k)[a:b] for k, _ in TX_FIELDS}
            sc = None if scores is None else scores[a - lo : b - lo]
            self.process(cols, out[a - lo : b - lo], scores=sc)  # écrit directement (memmap compris)
            if progress:
                el = time.perf_counter() - t0
                print(f"  {b - lo:>9,d}/{hi - lo:,d} tx  {(b - lo) / el:,.0f} tx/s", flush=True)
        return out

    def snapshot(self, path) -> int:
        """Écrit l'état du moteur (sans forêt) ; renvoie la taille en octets."""
        n = self.lib.aml_snapshot(self.h, str(path).encode())
        if n < 0:
            raise OSError(f"aml_snapshot a échoué (code {n})")
        return int(n)

    def restore(self, path) -> None:
        rc = self.lib.aml_restore(self.h, str(path).encode())
        if rc != 0:
            raise OSError(f"aml_restore a échoué (code {rc}) : instantané absent ou incompatible")

    def state_bytes(self) -> int:
        return int(self.lib.aml_state_bytes(self.h))

    def stats(self) -> dict[str, int]:
        n = self.lib.aml_n_stats()
        buf = (c_int64 * n)()
        self.lib.aml_stats(self.h, buf)
        return {self.lib.aml_stat_name(i).decode(): int(buf[i]) for i in range(n)}

    def close(self) -> None:
        if getattr(self, "h", None):
            self.lib.aml_free(self.h)
            self.h = None

    def __del__(self):
        self.close()
