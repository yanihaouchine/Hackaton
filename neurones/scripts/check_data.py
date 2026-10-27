"""Vérifie les données du MLP et ajuste la normalisation sur le train.
Usage (depuis neurones/) : ../.venv/bin/python scripts/check_data.py
Sorties : results/data_summary.json, results/preprocessing.json
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from nn.data import RESULTS, SEED, fit_preprocessing, load, rows_of, train_rows


def main() -> None:
    from aml.provenance import run_info  # date, commit, seed, machine (code de arbre/, lecture seule)

    t0 = time.perf_counter()
    ds = load()
    rows = train_rows(ds)
    x_tr, y_tr = rows_of(ds, rows)
    pre = fit_preprocessing(x_tr, [ds.names[i] for i in ds.cols])
    z = pre.transform(x_tr[:1000])

    splits = {}
    for name, (lo, hi) in ds.ranges.items():
        y = np.asarray(ds.y[lo:hi])
        splits[name] = {"rows": [lo, hi], "n_tx": hi - lo, "n_illicit": int(y.sum()), "rate": float(y.mean())}
    out = {
        "run": run_info(SEED),
        "features_digest": ds.features_digest,
        "n_features_engine": len(ds.names),
        "n_features_model": len(ds.cols),
        "n_inputs_network": pre.n_inputs,
        "one_hot": {pre.names[i]: k for i, k in zip(pre.cat_idx, pre.cat_sizes, strict=True)},
        "splits": splits,
        "train_sample": {"n_rows": int(len(rows)), "n_illicit": int(y_tr.sum()), "seed": SEED},
        "check_transformed": {"finite": bool(np.isfinite(z).all()), "shape": list(z.shape)},
        "seconds": round(time.perf_counter() - t0, 2),
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "data_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    pre.save(RESULTS / "preprocessing.json")
    print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
