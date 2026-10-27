"""Expériences sur la validation : baseline à règles, ablations par famille, variance entre seeds."""

from __future__ import annotations

import json

import numpy as np

from aml.config import EngineConfig
from aml.data import load, load_labels
from aml.features import load_features
from aml.provenance import write_result
from aml.rules import fit_rules
from aml.train import ABLATIONS, fit, select, subsample_rows


class Concat:
    """Vue paresseuse [X | H] (features strictes + rétrospectives J+1) indexable par lignes."""

    def __init__(self, a, b):
        self.a, self.b = a, b
        self.shape = (a.shape[0], a.shape[1] + b.shape[1])

    def __getitem__(self, rows):
        return np.hstack([np.asarray(self.a[rows]), np.asarray(self.b[rows])])


def _ctx(variant: str, cfg: EngineConfig | None = None, hindsight: bool = False, stage2: bool = False):
    s = load(variant)
    y = load_labels(variant)["y"]
    X, names, fmeta = load_features(variant, cfg)
    if stage2:
        from aml.features import feature_dir

        X = np.load(feature_dir(variant, cfg or EngineConfig()) / "X_stage2.npy", mmap_mode="r")
    if hindsight:
        from aml.features import load_hindsight

        H, hn = load_hindsight(variant, cfg)
        X, names = Concat(X, H), names + hn
    return s, y, X, names, fmeta


def run_rules(variant: str) -> dict:
    s, y, X, names, fmeta = _ctx(variant)
    lo, hi = s.split_range("val")
    chosen, res = fit_rules(np.asarray(X[lo:hi]), names, y[lo:hi])
    payload = {"name": "B0_regles", "variant": variant, "split": "val", "features": fmeta["config_digest"],
               "rules": chosen, "val": res}  # fmt: skip
    write_result("val_B0_regles", payload)
    return payload


def train_final(variant: str, name: str = "strict", families: list[str] | None = None,
                seeds=(42, 43, 44, 45, 46), neg_rate: float = 0.25, params: dict | None = None,
                hindsight: bool = False, stage2: bool = False,
                train_days: tuple[int, int] | None = None) -> dict:  # fmt: skip
    """Modèle livré = première seed ; les autres seeds mesurent la variance. Validation seulement."""
    from aml.evaluate import best_f1_threshold, binary_at, pr_auc
    from aml.forest import c_raw, from_lightgbm
    from aml.model import save_bundle, train_quantiles
    from aml.train import ALL_FAMILIES

    s, y, X, names, fmeta = _ctx(variant, hindsight=hindsight, stage2=stage2)
    families = (
        list(families or ALL_FAMILIES) + (["hindsight"] if hindsight else []) + (["propagation"] if stage2 else [])
    )
    cols = select(names, families)
    tr, va = s.split_range("train"), s.split_range("val")
    if train_days is not None:  # sous-période du train (validation croisée temporelle de l'étape 2)
        tr = (int(np.searchsorted(s.t, train_days[0] * 1440)), int(np.searchsorted(s.t, train_days[1] * 1440)))
    runs, kept = [], None
    for seed in seeds:
        rows = subsample_rows(y, *tr, neg_rate=neg_rate, seed=seed)
        booster, s_val, info = fit(X, y, names, rows, va, cols, seed=seed, params=params)
        params_used = info.pop("params")
        runs.append(info)
        print(f"{name} seed {seed}: PR-AUC {info['val_pr_auc']:.4f}  F1 {info['val_best_f1']['f1']:.4f}", flush=True)
        if kept is None:
            kept = (booster, s_val, info, params_used)
    booster, s_val, info, params_used = kept
    yv = y[va[0] : va[1]]
    thr = best_f1_threshold(yv, s_val)
    forest = from_lightgbm(booster, cols, names, booster.best_iteration,
                           meta={"name": name, "features": fmeta["config_digest"]})  # fmt: skip
    # contrôle : l'évaluateur C reproduit LightGBM sur la validation
    probe = np.asarray(X[va[0] : va[0] + 200_000], dtype=np.float32)
    raw_c = c_raw(forest, probe)
    raw_l = booster.predict(probe[:, cols], num_iteration=booster.best_iteration, raw_score=True)
    max_diff = float(np.max(np.abs(raw_c - raw_l)))
    if max_diff > 1e-6:
        raise RuntimeError(f"écart forêt C / LightGBM : {max_diff}")
    f1s = np.array([r["val_best_f1"]["f1"] for r in runs])
    aps = np.array([r["val_pr_auc"] for r in runs])
    model_info = {
        "name": name, "variant": variant, "families": families, "cols": cols,
        "feature_names": [names[i] for i in cols], "engine_features_digest": fmeta["config_digest"],
        "engine_config": fmeta["config"], "neg_rate": neg_rate, "seed": seeds[0],
        "best_iteration": int(booster.best_iteration), "threshold": thr["threshold"],
        "val_at_threshold": binary_at(yv, s_val, thr["threshold"]), "val_pr_auc": pr_auc(yv, s_val),
        "params": params_used, "c_vs_lightgbm_max_abs_diff": max_diff,
        "mode": "j1" if hindsight else "strict", "stage2": stage2, "train_days": train_days,
        "stage1": "strict" if stage2 else None,
    }  # fmt: skip
    q = train_quantiles(X, names, np.arange(*tr), max_rows=200_000)
    path = save_bundle(name, forest, model_info, q, booster)
    payload = {"model": model_info, "runs": runs, "model_dir": str(path.relative_to(path.parents[2])),
               "val_f1_mean": float(f1s.mean()), "val_f1_std": float(f1s.std()),
               "val_pr_auc_mean": float(aps.mean()), "val_pr_auc_std": float(aps.std())}  # fmt: skip
    write_result(f"val_model_{name}", payload)
    return payload


def run_ablations(variant: str, which: list[str] | None = None, seeds=(42,), neg_rate: float = 0.25) -> dict:
    s, y, X, names, fmeta = _ctx(variant)
    tr, va = s.split_range("train"), s.split_range("val")
    out = {}
    for name in which or list(ABLATIONS):
        cols = select(names, ABLATIONS[name])
        runs = []
        for seed in seeds:
            rows = subsample_rows(y, *tr, neg_rate=neg_rate, seed=seed)
            _, _, info = fit(X, y, names, rows, va, cols, seed=seed)
            info.pop("params")
            runs.append(info)
            print(f"{name:22s} seed {seed}: PR-AUC {info['val_pr_auc']:.4f}  F1 {info['val_best_f1']['f1']:.4f}  "
                  f"({info['best_iteration']} arbres, {info['train_seconds']} s)", flush=True)  # fmt: skip
        f1 = np.array([r["val_best_f1"]["f1"] for r in runs])
        ap = np.array([r["val_pr_auc"] for r in runs])
        res = {"name": name, "families": ABLATIONS[name], "n_features": len(cols), "neg_rate": neg_rate,
               "features": fmeta["config_digest"], "runs": runs,
               "val_f1_mean": float(f1.mean()), "val_f1_std": float(f1.std()),
               "val_pr_auc_mean": float(ap.mean()), "val_pr_auc_std": float(ap.std())}  # fmt: skip
        write_result(f"val_{name}", res)
        out[name] = res
    return out


def evaluate_model(variant: str, name: str, splits=("val",), unique: bool = True, cascade: bool = False) -> dict:
    """Évalue un modèle figé : seuil de validation appliqué tel quel ; le test n'est lu qu'ici.
    cascade=True applique le pré-filtrage tel qu'il fonctionne en ligne (transactions écartées : score du
    mini-modèle)."""
    from aml.evaluate import evaluate
    from aml.forest import c_raw
    from aml.model import load_bundle, load_cascade

    b = load_bundle(name)
    casc = load_cascade(b) if cascade else None
    if cascade and casc is None:
        raise FileNotFoundError("aucune cascade entraînée pour ce modèle (python -m aml cascade)")
    j1 = b.info.get("mode") == "j1"
    s, y, X, names, fmeta = _ctx(variant, hindsight=j1, stage2=bool(b.info.get("stage2")))
    if names != b.forest.engine_features:
        raise ValueError("le modèle ne correspond pas aux features calculées")
    lab = load_labels(variant)
    # début réel de chaque tentative (toutes périodes confondues) : le délai de détection en part
    att_all, t_all = lab["attempt"], np.asarray(s.t)
    m = att_all >= 0
    order = np.argsort(att_all[m], kind="stable")
    a_sorted, t_sorted = att_all[m][order], t_all[m][order]
    first = np.r_[0, np.flatnonzero(np.diff(a_sorted)) + 1]
    starts = {int(a_sorted[i]): int(np.min(t_sorted[i:j])) for i, j in zip(first, np.r_[first[1:], len(a_sorted)],
                                                                         strict=True)}  # fmt: skip
    out = {}
    for sp in splits:
        lo, hi = s.split_range(sp)
        xs = np.ascontiguousarray(X[lo:hi], dtype=np.float32)
        p = 1.0 / (1.0 + np.exp(-c_raw(b.forest, xs)))
        if casc is not None:
            r0 = c_raw(casc[0], xs)
            p = np.where(r0 >= casc[1], p, 1.0 / (1.0 + np.exp(-r0)))
        t = np.asarray(s.t[lo:hi])
        dec = (t // 1440 + 1) * 1440 if j1 else t
        res = evaluate(y[lo:hi], p, b.threshold, t, lab["attempt"][lo:hi], lab["typ"][lo:hi], decision_t=dec,
                       attempt_start=starts)  # fmt: skip
        # niveau cas : alertes regroupées (comptes partagés à moins de 96 h), priorité calibrée
        from aml.cases import build_cases
        from aml.evaluate import case_report

        al = np.flatnonzero(p >= b.threshold)
        ia = lo + al
        cases = build_cases(ia, np.asarray(s.src[ia]), np.asarray(s.dst[ia]), t[al],
                            np.asarray(s.usd_paid[ia], dtype=np.float64), b.calibrate(p[al]))  # fmt: skip
        res["cases"] = case_report(cases, lambda idx: y[np.asarray(idx)] == 1)
        out[sp] = res
        print(f"{name} [{sp}] P {res['precision']:.4f} R {res['recall']:.4f} F1 {res['f1']:.4f} "
              f"IC95 {res['f1_ci95'][0]:.3f}-{res['f1_ci95'][1]:.3f} PR-AUC {res['pr_auc']:.4f} "
              f"alertes {res['n_alerts']}", flush=True)  # fmt: skip
    payload = {"model": name, "variant": variant, "mode": b.info.get("mode", "strict"), "threshold": b.threshold,
               "cascade": cascade,
               "model_info": {k: b.info[k] for k in ("best_iteration", "neg_rate", "families", "forest_digest")
                              if k in b.info}, "splits": out}  # fmt: skip
    write_result(f"eval_{name}{'_cascade' if cascade else ''}_{'-'.join(splits)}", payload, unique=unique)
    return payload


HP_GRID = {
    "hp_base": {},
    "hp_neg100": {"_neg_rate": 1.0},
    "hp_leaves127": {"num_leaves": 127, "min_data_in_leaf": 100},
    "hp_leaves31": {"num_leaves": 31},
    "hp_ff05": {"feature_fraction": 0.5},
    "hp_lr003": {"learning_rate": 0.03},
}


def run_hparams(variant: str, which: list[str] | None = None, seed: int = 42) -> dict:
    """Petite recherche d'hyperparamètres sur la validation (modèle strict complet, une seed)."""
    from aml.train import ALL_FAMILIES

    s, y, X, names, fmeta = _ctx(variant)
    cols = select(names, ALL_FAMILIES)
    tr, va = s.split_range("train"), s.split_range("val")
    out = {}
    for name in which or list(HP_GRID):
        p = dict(HP_GRID[name])
        neg = p.pop("_neg_rate", 0.25)
        rows = subsample_rows(y, *tr, neg_rate=neg, seed=seed)
        _, _, info = fit(X, y, names, rows, va, cols, seed=seed, params=p)
        info.pop("params")
        print(f"{name:14s} PR-AUC {info['val_pr_auc']:.4f}  F1 {info['val_best_f1']['f1']:.4f}  "
              f"({info['best_iteration']} arbres, {info['train_seconds']} s)", flush=True)  # fmt: skip
        res = {"name": name, "params": p, "neg_rate": neg, **info}
        write_result(f"val_{name}", res)
        out[name] = res
    return out


def evaluate_explanations(variant: str, name: str = "strict", split: str = "val", max_alerts: int = 2000) -> dict:
    """Explicabilité mesurée sur les alertes d'une période : fidélité des 3 premières raisons, couverture des
    indicateurs à règle fixe, familles des raisons principales, accord avec la typologie connue."""
    from collections import Counter

    from aml.explain import contributions, faithfulness, family
    from aml.forest import c_raw
    from aml.model import load_bundle
    from aml.report import indicators

    b = load_bundle(name)
    s, y, X, names, _ = _ctx(variant, hindsight=b.info.get("mode") == "j1")
    lo, hi = s.split_range(split)
    xs = np.asarray(X[lo:hi], dtype=np.float32)
    p = 1.0 / (1.0 + np.exp(-c_raw(b.forest, xs)))
    idx = np.flatnonzero(p >= b.threshold)[:max_alerts]
    xa = xs[idx]
    contrib, method = contributions(b, xa)
    faith = faithfulness(b, xa, contrib, k=3)
    top_fam = Counter(family(names[int(np.argmax(c[: len(names)]))]) for c in contrib)
    cover = np.array([len(indicators(x, names)) > 0 for x in xa])
    yy = y[lo:hi][idx].astype(bool)
    res = {
        "model": name, "split": split, "method": method, "n_alerts": int(len(idx)), "faithfulness": faith,
        "indicator_coverage": float(cover.mean()), "indicator_coverage_true_alerts": float(cover[yy].mean()),
        "indicator_coverage_false_alerts": float(cover[~yy].mean()) if (~yy).any() else None,
        "top_reason_family": dict(top_fam.most_common()),
    }  # fmt: skip
    write_result(f"{split}_explain_{name}", res)
    return res


def train_cascade(variant: str, main: str = "strict", keep: float = 0.99, seed: int = 42) -> dict:
    """Cascade à la Viola-Jones : mini-forêt qui écarte les transactions manifestement licites avant la forêt
    complète. Seuil fixé sur la validation pour conserver `keep` des alertes du modèle complet."""
    from aml.evaluate import binary_at
    from aml.forest import c_raw, from_lightgbm
    from aml.model import load_bundle

    b = load_bundle(main)
    s, y, X, names, fmeta = _ctx(variant)
    cols = b.cols
    tr, va = s.split_range("train"), s.split_range("val")
    rows = subsample_rows(y, *tr, neg_rate=0.25, seed=seed)
    booster, _, info = fit(X, y, names, rows, va, cols, seed=seed, rounds=60, patience=20,
                           params={"num_leaves": 7, "learning_rate": 0.2, "min_data_in_leaf": 200})  # fmt: skip
    pre = from_lightgbm(booster, cols, names, booster.best_iteration, meta={"name": f"{main}_cascade"})
    xv = np.asarray(X[va[0] : va[1]], dtype=np.float32)
    yv = y[va[0] : va[1]]
    r_pre = c_raw(pre, xv)
    p_main = 1.0 / (1.0 + np.exp(-c_raw(b.forest, xv)))
    alerts = p_main >= b.threshold
    curve = []  # compromis complet, consigné pour la transparence (choix sur la validation)
    for k in (0.999, 0.995, 0.99, 0.98, 0.95):
        c = float(np.quantile(r_pre[alerts], 1.0 - k))
        ok = r_pre >= c
        pc = np.where(ok, p_main, 1.0 / (1.0 + np.exp(-r_pre)))
        curve.append({"keep": k, "cut_raw": c, "skip_rate": float(1.0 - ok.mean()),
                      "alerts_lost": int((alerts & ~ok).sum()),
                      "f1": binary_at(yv, pc, b.threshold)["f1"]})  # fmt: skip
    cut = float(np.quantile(r_pre[alerts], 1.0 - keep))
    passed = r_pre >= cut
    p_casc = np.where(passed, p_main, 1.0 / (1.0 + np.exp(-r_pre)))
    res = {
        "main": main, "cut_raw": cut, "keep_target": keep, "n_trees": pre.n_trees, "n_nodes": pre.n_nodes,
        "val_skip_rate": float(1.0 - passed.mean()), "val_alerts_main": int(alerts.sum()),
        "val_alerts_lost": int((alerts & ~passed).sum()),
        "val_true_alerts_lost": int((alerts & ~passed & (yv == 1)).sum()),
        "val_main": binary_at(yv, p_main, b.threshold), "val_cascade": binary_at(yv, p_casc, b.threshold),
        "pre_train": {k: info[k] for k in ("best_iteration", "val_pr_auc")}, "curve": curve,
    }  # fmt: skip
    size = pre.save(b.path / "cascade.amlz")
    res["cascade_bytes"] = size
    (b.path / "cascade.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    write_result(f"val_cascade_{main}", res)
    return res


def pareto(variant: str, name: str = "strict", ks=(50, 100, 200, 300, 500, 800, None)) -> dict:
    """Front de Pareto taille/qualité : modèle tronqué à k arbres (seuil re-choisi sur la validation),
    coût d'évaluation mesuré avec l'évaluateur C sur 50 000 transactions de validation."""
    import time

    import lightgbm as lgb

    from aml.evaluate import best_f1_threshold, pr_auc
    from aml.forest import c_raw, from_lightgbm
    from aml.model import load_bundle

    b = load_bundle(name)
    s, y, X, names, _ = _ctx(variant, hindsight=b.info.get("mode") == "j1", stage2=bool(b.info.get("stage2")))
    booster = lgb.Booster(model_file=str(b.lgbm_path()))
    lo, hi = s.split_range("val")
    xv = np.ascontiguousarray(X[lo:hi], dtype=np.float32)
    yv = y[lo:hi]
    probe = xv[:50_000]
    rows = []
    for k in ks:
        f = from_lightgbm(booster, b.cols, names, num_iteration=k)
        p = 1.0 / (1.0 + np.exp(-c_raw(f, xv)))
        t0 = time.perf_counter()
        c_raw(f, probe)
        us = (time.perf_counter() - t0) / len(probe) * 1e6
        bf = best_f1_threshold(yv, p)
        rows.append({"trees": f.n_trees, "nodes": f.n_nodes, "val_pr_auc": pr_auc(yv, p), "val_best_f1": bf["f1"],
                     "eval_us_per_tx": us, "bytes_compact": int(f.n_nodes * 12 + f.n_nodes * 8)})  # fmt: skip
        r = rows[-1]
        print(f"{f.n_trees:5d} arbres : PR-AUC {r['val_pr_auc']:.4f} F1 {bf['f1']:.4f} {us:.2f} µs/tx", flush=True)
    res = {"model": name, "rows": rows, "note": "coûts mesurés sur la machine courante, un thread"}
    write_result(f"val_pareto_{name}", res)
    return res


def motif_formats_ablation(variant: str, formats: list[str], ring_bits: int = 22, seed: int = 42) -> dict:
    """Graphe de motifs étendu à d'autres formats que l'ACH : coût (débit, état) contre qualité (validation)."""
    from aml.features import compute, load_features
    from aml.train import ALL_FAMILIES

    cfg = EngineConfig(motif_formats=formats, ring_bits=ring_bits)
    d = compute(variant, cfg, progress=False)
    fmeta = json.loads((d / "meta.json").read_text())
    s = load(variant)
    y = load_labels(variant)["y"]
    X, names, _ = load_features(variant, cfg)
    cols = select(names, ALL_FAMILIES)
    tr, va = s.split_range("train"), s.split_range("val")
    rows = subsample_rows(y, *tr, neg_rate=0.25, seed=seed)
    _, _, info = fit(X, y, names, rows, va, cols, seed=seed)
    info.pop("params")
    res = {"motif_formats": formats, "ring_bits": ring_bits, "engine_tx_per_s": fmeta["tx_per_s"],
           "engine_state_bytes": fmeta["state_bytes"], "engine_stats": fmeta["stats"], **info}  # fmt: skip
    write_result("val_motifs_tous_formats", res)
    return res
