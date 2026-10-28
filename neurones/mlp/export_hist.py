"""Export ONNX du modèle séquentiel livré, vérification contre torch, et frugalité (même protocole que mlp.bench).

Entrées du graphe : features brutes (96 colonnes du moteur), séquences émetteur et destinataire (K × EVENT_DIM),
masques de validité (K, en float). Sortie : probabilité recalibrée.
Mesures séparées : (1) le modèle seul, sur des séquences déjà préparées ; (2) la préparation des séquences à
partir de l'index d'historique, dans ce prototype Python (numpy). L'état d'historique en flux est dimensionné,
pas mesuré : il n'est pas encore implémenté en flux.
Usage (depuis neurones/) : python -m mlp.export_hist [--model hist_h64-32_g16_s42]
Sortie : results/export_hist_<machine>.json
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from mlp.bench import BATCH, N_BATCH, N_SINGLE, max_rss_mb, pct, timed
from mlp.common import ARTIFACTS, write_result
from mlp.history import EVENT_DIM, K, batch
from mlp.train_hist import HistScorer


class Exported(nn.Module):
    """Modèle livré : (features, séquences, masques) → probabilité recalibrée."""

    def __init__(self, net, offset: float):
        super().__init__()
        self.net, self.offset = net, offset

    def forward(self, x, es, ms, ed, md):
        return torch.sigmoid(self.net(x, es, ms, ed, md) + self.offset)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="hist_h64-32_g16_s42")
    a = p.parse_args()

    sc = HistScorer(a.model)
    model = Exported(sc.net.eval(), sc.offset).eval()
    lo, _ = sc.stream.split_range("val")
    rows = np.arange(lo, lo + N_BATCH)
    x = np.ascontiguousarray(sc.X[rows][:, sc.cols], dtype=np.float32)
    t = time.perf_counter()
    es, ms, ed, md = batch(*sc.hist, rows, sc.ram)
    prep_batch_s = time.perf_counter() - t
    ms, md = ms.astype(np.float32), md.astype(np.float32)
    feeds = {"features": x, "seq_src": es, "mask_src": ms, "seq_dst": ed, "mask_dst": md}

    path = ARTIFACTS / a.model / "model.onnx"
    example = tuple(torch.from_numpy(v[:2]) for v in feeds.values())
    torch.onnx.export(model, example, str(path), input_names=list(feeds), output_names=["score"],
                      dynamic_axes={**{k: {0: "n"} for k in feeds}, "score": {0: "n"}}, dynamo=False)  # fmt: skip

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    t = time.perf_counter()
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t) * 1e3
    n_check = 20_000
    s_ort = sess.run(None, {k: v[:n_check] for k, v in feeds.items()})[0]
    with torch.no_grad():
        s_torch = model(*(torch.from_numpy(v[:n_check]) for v in feeds.values())).numpy()
    diff = float(np.max(np.abs(s_ort - s_torch)))
    # Avertissement de torch sur les GRU exportés avec un lot > 1 : on vérifie les petits lots et l'invariance au lot
    small = {}
    for n in (1, 3):
        f = {k: v[:n] for k, v in feeds.items()}
        with torch.no_grad():
            ref = model(*(torch.from_numpy(v) for v in f.values())).numpy()
        small[str(n)] = float(np.max(np.abs(sess.run(None, f)[0] - ref)))
    one = np.array([sess.run(None, {k: v[i : i + 1] for k, v in feeds.items()})[0][0] for i in range(0, 4096, 64)])
    small["one_by_one_vs_batch"] = float(np.max(np.abs(one - s_ort[:4096:64])))

    def run(i0: int, i1: int):
        return sess.run(None, {k: v[i0:i1] for k, v in feeds.items()})

    # Latence et débit du modèle seul (séquences déjà préparées), protocole de mlp.bench
    for i in range(50):
        run(i, i + 1)
    lat = []
    for i in range(N_SINGLE):
        t = time.perf_counter()
        run(i, i + 1)
        lat.append((time.perf_counter() - t) * 1e6)
    t = time.perf_counter()
    for i in range(0, N_BATCH, BATCH):
        run(i, i + BATCH)
    model_bench = {"latency_us_p50": pct(lat, 50), "latency_us_p99": pct(lat, 99),
                   "throughput_tx_s": round(N_BATCH / (time.perf_counter() - t))}  # fmt: skip

    # Préparation des séquences (prototype numpy) : par transaction et par lot
    prep_one = timed(lambda r: batch(*sc.hist, r, sc.ram), rows)
    n_acc = int(max(sc.ram.src.max(), sc.ram.dst.max())) + 1
    event_bytes = 4 + 4 + 1 + 4 + 1  # instant (int32), montant (float32), format, contrepartie (int32), drapeaux
    payload = {
        "model": a.model,
        "threads": 1,
        "onnx": {
            "path": str(path.relative_to(ARTIFACTS.parent)),
            "file_bytes": path.stat().st_size,
            "load_ms": round(load_ms, 1),
            "max_abs_diff_ort_torch": diff,
            "n_checked": n_check,
            "small_batches_max_abs_diff": small,
        },  # fmt: skip
        "n_params": int(sum(q.numel() for q in sc.net.parameters())),
        "model_only": model_bench,
        "sequence_preparation_numpy": {
            **prep_one,
            "batch_of_200k_s": round(prep_batch_s, 2),
            "note": "prototype Python : index d'historique + numpy ; pas le coût d'une implémentation en flux",
        },  # fmt: skip
        "history_state_sizing": {
            "accounts": n_acc,
            "k": K,
            "bytes_per_event": event_bytes,
            "bytes": n_acc * K * event_bytes,
            "measured": False,
            "note": "tampon circulaire par compte, non implémenté",
        },  # fmt: skip
        "event_dim": EVENT_DIM,
        "max_rss_mb_process": max_rss_mb(),
        "energy": "non mesuré",
    }
    out = write_result(f"export_hist_{platform.machine()}", payload, seed=42)
    m = payload["onnx"]
    print(f"ONNX {m['file_bytes'] / 1024:.0f} Ko, écart torch {diff:.2e} ; modèle seul : p50 "
          f"{model_bench['latency_us_p50']} µs, p99 {model_bench['latency_us_p99']} µs, "
          f"{model_bench['throughput_tx_s']:,} tx/s")  # fmt: skip
    print(f"préparation des séquences (numpy) : p50 {prep_one['latency_us_p50']} µs par transaction, "
          f"{prep_one['throughput_tx_s']:,} tx/s en lots ; état d'historique dimensionné "
          f"{payload['history_state_sizing']['bytes'] / 2**20:.0f} Mo -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()
