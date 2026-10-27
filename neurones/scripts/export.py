"""Exporte le MLP retenu en ONNX (normalisation comprise) et vérifie qu'il donne les mêmes scores que torch.
Usage (depuis neurones/) : ../.venv/bin/python scripts/export.py
Sorties : artifacts/mlp.onnx (régénérable, hors git), results/export_mlp.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from nn.data import RESULTS, Preprocessing, load, rows_of
from nn.model import Scorer, mlp, n_params

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
N_CHECK = 20_000


def load_scorer() -> tuple[Scorer, dict]:
    ckpt = torch.load(ARTIFACTS / "mlp.pt", weights_only=False)
    pre = Preprocessing(**ckpt["preprocessing"])
    net = mlp(pre.n_inputs)
    net.load_state_dict(ckpt["state_dict"])
    scorer = Scorer(pre, net).eval()
    return scorer, ckpt


def main() -> None:
    import onnxruntime as ort
    from aml.provenance import run_info

    scorer, ckpt = load_scorer()
    path = ARTIFACTS / "mlp.onnx"
    dummy = torch.zeros(2, len(scorer.norm.mean) + len(scorer.norm.cats))
    torch.onnx.export(scorer, (dummy,), str(path), input_names=["features"], output_names=["proba"],
                      dynamic_axes={"features": {0: "n"}, "proba": {0: "n"}}, opset_version=17,
                      dynamo=False)  # fmt: skip

    # Vérification sur des transactions de validation : torch (référence) contre onnxruntime (livré)
    ds = load()
    lo, _ = ds.ranges["val"]
    x, _ = rows_of(ds, (lo, lo + N_CHECK))
    with torch.no_grad():
        p_torch = scorer(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx = sess.run(None, {"features": x})[0]
    diff = float(np.max(np.abs(p_torch - p_onnx)))
    thr_logit = float(ckpt["threshold_logit"])
    out = {
        "run": run_info(),
        "onnx_bytes": path.stat().st_size,
        "n_params": n_params(scorer.net),
        "opset": 17,
        "n_inputs_raw": int(x.shape[1]),
        "threshold_logit": thr_logit,
        "threshold_proba": float(1 / (1 + np.exp(-thr_logit))),
        "check_rows": N_CHECK,
        "max_abs_diff_torch_vs_onnx": diff,
    }
    assert diff < 1e-5, f"écart torch / ONNX trop grand : {diff}"
    (RESULTS / "export_mlp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=2))


if __name__ == "__main__":
    main()
