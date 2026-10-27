"""Fidélité des raisons du MLP sur la VALIDATION (même protocole que `python -m aml explain-eval` de arbre/),
et exemple de raisons pour l'alerte la plus forte.
Usage (depuis neurones/) : python -m mlp.explain_eval [--model h64-32_n100_s42]  ->  results/explain_mlp.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mlp.common import REPO, load_features, load_stream, write_result
from mlp.explain import faithfulness, reasons
from mlp.train import load_scorer

FOREST_RESULT = REPO / "arbre" / "results" / "val_explain_strict.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    a = p.parse_args()
    from aml.model import load_bundle

    scorer, meta = load_scorer(a.model)
    thr = meta["threshold"]
    X, all_names, cols = load_features()
    stream, _ = load_stream()
    bundle = load_bundle("strict")  # noms, percentiles du train et médianes : features communes
    names = [all_names[i] for i in cols]
    median = np.array([bundle.quantiles[n][4] for n in names], dtype=np.float32)
    lo, hi = stream.split_range("val")
    x = np.asarray(X[lo:hi][:, cols], dtype=np.float32)

    faith = {k: faithfulness(scorer, x, median, thr, k=k) for k in (1, 3, 5)}

    with torch.no_grad():
        s = scorer(torch.from_numpy(x)).numpy()
    i = int(np.argmax(s))
    rs = reasons(scorer, bundle, np.asarray(X[lo + i]), cols, stream.meta["countries"])
    keys = ("feature", "text", "contribution", "percentile")
    example = {"tx_index": lo + i, "score": float(s[i]), "reasons": [{k: r[k] for k in keys} for r in rs]}

    forest = json.loads(FOREST_RESULT.read_text(encoding="utf-8")) if FOREST_RESULT.exists() else None
    write_result("explain_mlp", {"model": a.model, "split": "val",
                                 "method": "gradient × entrée (logit), entrées d'une même feature regroupées",
                                 "faithfulness": faith, "example": example, "forest_explain_val": forest},
                 seed=42)  # fmt: skip
    print(json.dumps({"faithfulness": faith, "example": example}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
