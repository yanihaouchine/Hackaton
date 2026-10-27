"""Portabilité : le pilote C compilé pour aarch64 (exécuté sous qemu) donne les mêmes features qu'en x86-64.

Sous qemu, on vérifie la portabilité, jamais les performances. Le test est ignoré si zig ou qemu manquent
(qemu-aarch64 : voir tools/qemu, installé sans droits root par apt-get download + dpkg -x).
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import numpy as np
import pytest

from aml.config import BUILD, DATA_PROC, REPO, ROOT, EngineConfig

QEMU = REPO / "tools" / "qemu" / "extracted" / "usr" / "bin" / "qemu-aarch64"
SRC = [str(ROOT / "tests" / "engine_main.c"), str(ROOT / "aml" / "engine" / "aml_engine.c")]
FLAGS = ["-O2", "-std=c99", "-ffp-contract=off"]

pytestmark = pytest.mark.skipif(
    not (DATA_PROC / "HI-Small" / "meta.json").exists() or not QEMU.exists(), reason="données ou qemu absents"
)


def _dump(path, s, cols, cfg: EngineConfig) -> None:
    from aml.config import FORMATS

    conf = np.array([len(s.acc_etype), cfg.win_short, cfg.win_mid, cfg.win_long, cfg.fmt_mask(),
                     FORMATS.index("ACH"), cfg.ring_bits, cfg.pair_bits, cfg.pair_ttl, cfg.scan_cap,
                     cfg.fan_cap, cfg.cycle_max_len, cfg.cycle_budget], dtype=np.int32)  # fmt: skip
    with open(path, "wb") as fh:
        fh.write(conf.tobytes() + np.float32(cfg.chain_tol).tobytes())
        fh.write(np.int64(len(cols["t"])).tobytes())
        for k, dt in (("acc_bank", np.int32), ("acc_entity", np.int32), ("acc_etype", np.uint8),
                      ("acc_nacc", np.int32), ("acc_country", np.int16)):  # fmt: skip
            fh.write(np.ascontiguousarray(getattr(s, k), dtype=dt).tobytes())
        for k, dt in (("t", np.int32), ("src", np.int32), ("dst", np.int32), ("usd_paid", np.float32),
                      ("usd_recv", np.float32), ("amt_paid", np.float64), ("cur_paid", np.uint8),
                      ("cur_recv", np.uint8), ("fmt", np.uint8)):  # fmt: skip
            fh.write(np.ascontiguousarray(cols[k], dtype=dt).tobytes())


def _read(path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    nf = int(raw[:4].view(np.int32)[0])
    return raw[4:].view(np.float32).reshape(-1, nf)


def test_aarch64_matches_x86(tmp_path):
    from aml.data import load
    from aml.engine import TX_FIELDS

    if shutil.which("gcc") is None:
        pytest.skip("gcc absent")
    s = load("HI-Small")
    lo, hi = np.searchsorted(s.t, 1440), np.searchsorted(s.t, 2 * 1440)
    fmt = np.asarray(s.fmt[lo:hi])
    idx = lo + np.flatnonzero(fmt == 0)[:40_000]
    cols = {k: np.asarray(getattr(s, k)[idx]) for k, _ in TX_FIELDS}
    _dump(tmp_path / "in.bin", s, cols, EngineConfig())
    x86 = BUILD / "engine_main-x86_64"
    arm = BUILD / "engine_main-aarch64"
    BUILD.mkdir(exist_ok=True)
    subprocess.run(["gcc", *FLAGS, *SRC, "-lm", "-o", str(x86)], check=True)
    subprocess.run([sys.executable, "-m", "ziglang", "cc", "-target", "aarch64-linux-musl", "-mcpu=baseline",
                    "-static", *FLAGS, *SRC, "-lm", "-o", str(arm)], check=True)  # fmt: skip
    subprocess.run([str(x86), str(tmp_path / "in.bin"), str(tmp_path / "x86.bin")], check=True)
    subprocess.run([str(QEMU), str(arm), str(tmp_path / "in.bin"), str(tmp_path / "arm.bin")], check=True)
    a, b = _read(tmp_path / "x86.bin"), _read(tmp_path / "arm.bin")
    assert a.shape == b.shape
    same = float(np.mean(a == b))
    assert np.allclose(a, b, rtol=1e-5, atol=1e-5), f"écart max {np.max(np.abs(a - b))}"
    print(f"\nfeatures identiques bit à bit : {100 * same:.3f} %")
