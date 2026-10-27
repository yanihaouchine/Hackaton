"""Fidélité des raisons du MLP sur la VALIDATION (même protocole que `python -m aml explain-eval` de arbre/),
et exemple de raisons pour l'alerte la plus forte.
Usage (depuis neurones/) : ../.venv/bin/python scripts/explain_eval.py
Sortie : results/explain_mlp.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from nn.data import ARBRE, RESULTS, load, rows_of
from nn.explain import faithfulness, reasons

FOREST_RESULT = ARBRE / "results" / "val_explain_strict.json"


def main() -> None:
    from aml.data import load as load_stream
    from aml.model import load_bundle
    from aml.provenance import run_info

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export import load_scorer

    scorer, _ = load_scorer()
    export = json.loads((RESULTS / "export_mlp.json").read_text(encoding="utf-8"))
    thr = export["threshold_proba"]
    ds = load()
    bundle = load_bundle("strict")  # noms, percentiles du train et médianes : features communes
    names = [ds.names[i] for i in ds.cols]
    median = np.array([bundle.quantiles[n][4] for n in names], dtype=np.float32)
    x, _ = rows_of(ds, ds.ranges["val"])

    faith = {k: faithfulness(scorer, x, median, thr, k=k) for k in (1, 3, 5)}

    with torch.no_grad():
        p = scorer(torch.from_numpy(x)).numpy()
    i = int(np.argmax(p))
    lo, _ = ds.ranges["val"]
    countries = load_stream("HI-Small").meta["countries"]
    rs = reasons(scorer, bundle, np.asarray(ds.X[lo + i]), ds.cols, countries)
    keys = ("feature", "text", "contribution", "percentile")
    example = {"tx_index": lo + i, "proba": float(p[i]), "reasons": [{k: r[k] for k in keys} for r in rs]}

    forest = json.loads(FOREST_RESULT.read_text(encoding="utf-8")) if FOREST_RESULT.exists() else None
    out = {"run": run_info(), "split": "val", "method": "gradient × entrée (logit), one-hot regroupés",
           "faithfulness": faith, "example": example, "forest_explain_val": forest}  # fmt: skip
    (RESULTS / "explain_mlp.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"faithfulness": faith, "example": example}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
