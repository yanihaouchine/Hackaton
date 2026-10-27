"""Tableaux Markdown construits uniquement à partir de results/*.json (chiffres traçables pour README et slides)."""

from __future__ import annotations

import json
from pathlib import Path

from aml.config import RESULTS
from aml.train import ABLATIONS


def _load(pattern: str) -> list[tuple[Path, dict]]:
    return [(p, json.loads(p.read_text())) for p in sorted(RESULTS.glob(pattern))]


def _f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}".replace(".", ",")


def ablations() -> str:
    rows = ["| Configuration | Familles | PR-AUC val | F1 val (meilleur seuil) | Seeds |", "|---|---|---|---|---|"]
    b0 = _load("val_B0_regles.json")
    if b0:
        v = b0[0][1]["val"]
        rows.append(f"| B0 règles expertes | règles seuillées | — | {_f(v['f1'])} | — |")
    for name in ABLATIONS:
        r = _load(f"val_{name}.json")
        if not r:
            continue
        d = r[0][1]
        n = len(d["runs"])
        std = f" ± {_f(d['val_f1_std'])}" if n > 1 else ""
        rows.append(f"| {name} | {', '.join(d['families'])} | {_f(d['val_pr_auc_mean'])} | "
                    f"{_f(d['val_f1_mean'])}{std} | {n} |")  # fmt: skip
    return "\n".join(rows)


def models() -> str:
    rows = ["| Modèle | Mode | PR-AUC val (moy. ± é.-t.) | F1 val (moy. ± é.-t.) | Seuil | Arbres | Forêt (octets) |",
            "|---|---|---|---|---|---|---|"]  # fmt: skip
    for _, d in _load("val_model_*.json"):
        m = d["model"]
        rows.append(f"| {m['name']} | {m.get('mode', 'strict')}{' + étape 2' if m.get('stage2') else ''} | "
                    f"{_f(d['val_pr_auc_mean'])} ± {_f(d['val_pr_auc_std'])} | {_f(d['val_f1_mean'])} ± "
                    f"{_f(d['val_f1_std'])} | {_f(m['threshold'], 4)} | {m.get('best_iteration')} | — |")  # fmt: skip
    return "\n".join(rows)


def evaluations() -> str:
    rows = [
        "| Fichier | Modèle | Période | P | R | F1 [IC95] | PR-AUC | Alertes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p, d in _load("eval_*.json"):
        for sp, r in d.get("splits", {}).items():
            ci = r.get("f1_ci95", [None, None])
            rows.append(f"| {p.name} | {d['model']} | {sp} | {_f(r['precision'])} | {_f(r['recall'])} | "
                        f"{_f(r['f1'])} [{_f(ci[0])}–{_f(ci[1])}] | {_f(r['pr_auc'])} | {r['n_alerts']} |")  # fmt: skip
    return "\n".join(rows)


def typologies(eval_file: str, split: str) -> str:
    d = json.loads((RESULTS / eval_file).read_text())["splits"][split]["typology"]
    rows = ["| Typologie | Tentatives | Détectées | Rappel tx | Délai médian (h) |", "|---|---|---|---|---|"]
    for k, v in d.items():
        if k.startswith("_"):
            continue
        rows.append(f"| {k} | {v['attempts']} | {v['attempts_detected']} | {_f(v['tx_recall'])} | "
                    f"{_f(v['median_delay_h'], 1)} |")  # fmt: skip
    rows.append(f"| hors motifs (tx) | — | — | {_f(d['_non_pattern']['tx_recall'])} | — |")
    return "\n".join(rows)


def bench() -> str:
    rows = ["| Machine | Débit (tx/s) | Latence p50 / p99 (µs) | État moteur (Mio) | RSS max (Mio) | Énergie (µJ/tx) |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for p, d in _load("bench_*.json"):
        lat = d["latency_engine_ns"]
        mem = d["memory_mib"]
        e = d["energy"]
        rows.append(f"| {d['run']['machine']['cpu']} ({p.stem}) | {d['stream']['tx_per_s']:,.0f} | "
                    f"{lat['p50'] / 1e3:.1f} / {lat['p99'] / 1e3:.1f} | {mem['engine_state']:.0f} | "
                    f"{mem.get('VmHWM', mem['ru_maxrss']):.0f} | "
                    f"{_f(e.get('uj_per_tx'), 2) if 'uj_per_tx' in e else e['methode']} |"
                    .replace(",", "\u202f", 1))  # fmt: skip
    return "\n".join(rows)


# F1 de la classe illicite publiés sur HI-Small (protocole par jours de Multi-GNN, ou 60/20/20 pour GFP)
LITERATURE = [
    ("LightGBM, features de base", "GFP, ICAIF'24 (arXiv 2402.08593)", 21.30, "non"),
    ("GFP + XGBoost, lots de 128", "GFP, ICAIF'24", 63.23, "contenu du lot"),
    ("GFP + XGBoost, lots de 2 048", "GFP, ICAIF'24", 64.77, "contenu du lot"),
    ("Multi-PNA", "MEGA-GNN (arXiv 2412.00241)", 68.16, "tout le graphe de test"),
    ("FraudGT", "SALT-GNN (alphaxiv 2607.10131)", 67.9, "tout le graphe de test"),
    ("SALT-Trans", "SALT-GNN", 71.6, "tout le graphe de test"),
    ("MEGA-GenAgg", "MEGA-GNN", 74.88, "tout le graphe de test"),
]


LITERATURE_LI = [
    ("GFP + XGBoost, lots de 128", "GFP, ICAIF'24", 27.28, "contenu du lot"),
    ("FraudGT", "BlazingAML (arXiv 2604.12241)", 28.6, "tout le graphe de test"),
    ("Multi-PNA", "MEGA-GNN (arXiv 2412.00241)", 33.07, "tout le graphe de test"),
    ("MEGA-GenAgg", "MEGA-GNN", 46.29, "tout le graphe de test"),
]


def _variant(model: str) -> str:
    from aml.config import ARTIFACTS

    f = ARTIFACTS / "models" / model / "model.json"
    return json.loads(f.read_text()).get("variant", "HI-Small") if f.exists() else "HI-Small"


def sota(variant: str = "HI-Small") -> str:
    rows = ["| Méthode | Source | F1 (%) | Accès au futur |", "|---|---|---|---|"]
    for name, src, f1, fut in LITERATURE if variant == "HI-Small" else LITERATURE_LI:
        rows.append(f"| {name} | {src} | {_f(f1, 1)} | {fut} |")
    for p, d in _load("eval_*test*.json"):
        r = d.get("splits", {}).get("test_lit")
        if r and _variant(d["model"]) == variant and not d.get("cascade"):
            mode = "aucun (strict)" if d.get("mode") == "strict" else "jusqu'à minuit (revue J+1)"
            rows.append(f"| **Ce travail : {d['model']}** | `{p.name}` | **{_f(100 * r['f1'], 1)}** | {mode} |")
    return "\n".join(rows)


def llm() -> str:
    rows = [
        "| Fichier | Modèle | Acceptation (garde) | Latence médiane (s) | Chargement (s) |",
        "|---|---|---|---|---|",
    ]
    for p, d in _load("llm_bench_*.json"):
        for m, r in d["models"].items():
            rows.append(f"| {p.stem} | {m} | {_f(100 * r['acceptance_rate'], 0)} % | {_f(r['latency_s_median'], 1)} | "
                        f"{_f(r['load_s'], 1)} |")  # fmt: skip
    return "\n".join(rows)


def main() -> None:
    for title, fn in (("Ablations (validation)", ablations), ("Modèles livrés (validation)", models),
                      ("Évaluations", evaluations), ("Comparaison à l'état de l'art, HI-Small", sota),
                      ("Comparaison à l'état de l'art, LI-Small", lambda: sota("LI-Small")),
                      ("Frugalité", bench), ("LLM (rédaction sous garde)", llm)):  # fmt: skip
        print(f"\n### {title}\n")
        print(fn())
