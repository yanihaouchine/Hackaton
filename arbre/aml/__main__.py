"""Point d'entrée : python -m aml <commande> [options]."""

from __future__ import annotations

import argparse
import json
import sys

from aml.config import DEFAULT_VARIANT


def _cmd_prepare(a: argparse.Namespace) -> None:
    from aml.data import prepare

    out = prepare(a.variant)
    print(json.dumps(json.loads((out / "meta.json").read_text()), indent=2, ensure_ascii=False))


def _cmd_features(a: argparse.Namespace) -> None:
    from aml.config import EngineConfig
    from aml.features import compute

    cfg = EngineConfig()
    if a.motif_formats:
        cfg.motif_formats = a.motif_formats.split(",")
    if a.ring_bits:
        cfg.ring_bits = a.ring_bits
    out = compute(a.variant, cfg)
    if a.hindsight:
        from aml.features import compute_hindsight

        compute_hindsight(a.variant, cfg, progress=False)
    meta = json.loads((out / "meta.json").read_text())
    meta.pop("feature_names")
    print(json.dumps(meta, indent=2))


def _cmd_rules(a: argparse.Namespace) -> None:
    from aml.experiments import run_rules

    print(json.dumps(run_rules(a.variant), indent=2, ensure_ascii=False))


def _cmd_ablate(a: argparse.Namespace) -> None:
    from aml.experiments import run_ablations

    which = a.only.split(",") if a.only else None
    seeds = tuple(int(x) for x in a.seeds.split(","))
    run_ablations(a.variant, which, seeds=seeds, neg_rate=a.neg_rate)


def _cmd_train(a: argparse.Namespace) -> None:
    from aml.experiments import train_final

    seeds = tuple(int(x) for x in a.seeds.split(","))
    fam = a.families.split(",") if a.families else None
    days = tuple(int(x) for x in a.train_days.split(",")) if a.train_days else None
    res = train_final(a.variant, a.name, fam, seeds=seeds, neg_rate=a.neg_rate, hindsight=a.hindsight,
                      stage2=a.stage2, train_days=days)  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k != "runs"}, indent=2, ensure_ascii=False)[:3000])


def _cmd_export_lgbm(a: argparse.Namespace) -> None:
    from aml.experiments import export_lgbm

    for name in a.models.split(","):
        print(json.dumps(export_lgbm(a.variant, name), ensure_ascii=False), flush=True)


def _cmd_evaluate(a: argparse.Namespace) -> None:
    from aml.experiments import evaluate_model

    evaluate_model(a.variant, a.model, tuple(a.splits.split(",")), unique="val" != a.splits, cascade=a.cascade)


def _cmd_transfer(a: argparse.Namespace) -> None:
    from aml.transfer import THREADS, run

    run(a.variant, a.model, cascade=not a.no_cascade, threads=a.threads or THREADS)


def _cmd_typology(a: argparse.Namespace) -> None:
    from aml.typology import train

    print(json.dumps(train(a.variant), indent=2, ensure_ascii=False))


def _cmd_bench(a: argparse.Namespace) -> None:
    from aml.bench import run

    res = run(a.variant, a.model, watts_idle=a.watts_idle, watts_load=a.watts_load, tag=a.tag,
              cascade=not a.no_cascade)  # fmt: skip
    print(json.dumps({k: res[k] for k in ("stream", "latency_engine_ns", "latency_python_call_us", "memory_mib",
                                          "sizes_bytes", "energy")}, indent=2, ensure_ascii=False))  # fmt: skip


def _cmd_demo(a: argparse.Namespace) -> None:
    from aml.demo import run

    run(a.variant, a.start, a.days, a.reports, a.llm, a.model)


def _cmd_ask(a: argparse.Namespace) -> None:
    from aml.assistant import ask
    from aml.config import OUT

    f = json.loads((OUT / "demo" / f"{a.case}.json").read_text())
    llm = None
    if a.llm:
        from aml.llm import LLM

        llm = LLM()
    r = ask(f, a.question, llm)
    print(f"[{r['source']}]\n{r['text']}")


def _cmd_explain_eval(a: argparse.Namespace) -> None:
    from aml.experiments import evaluate_explanations

    print(json.dumps(evaluate_explanations(a.variant, a.model, a.split), indent=2, ensure_ascii=False))


def _cmd_stage2_features(a: argparse.Namespace) -> None:
    from aml.features import compute_stage2

    print(compute_stage2(a.variant))


def _cmd_summarize(a: argparse.Namespace) -> None:
    from aml.summarize import main as summarize

    summarize()


def _cmd_cascade(a: argparse.Namespace) -> None:
    from aml.experiments import train_cascade

    print(json.dumps(train_cascade(a.variant, a.model), indent=2, ensure_ascii=False))


def _cmd_llm_bench(a: argparse.Namespace) -> None:
    from aml.config import OUT
    from aml.llm import bench
    from aml.provenance import write_result

    files = sorted((OUT / "demo").glob("C*.json"))[: a.n]
    res = bench([json.loads(f.read_text()) for f in files], n_threads=a.threads)
    import platform

    write_result(f"llm_bench_{platform.node()}_{platform.machine()}", res)
    print(json.dumps(res, indent=2, ensure_ascii=False)[:4000])


def _cmd_pareto(a: argparse.Namespace) -> None:
    from aml.experiments import pareto

    pareto(a.variant, a.model)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="aml", description="Détection LCB-FT en flux (IBM AML)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prepare", help="CSV → flux encodé trié (data/processed/<variante>)")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.set_defaults(func=_cmd_prepare)

    s = sub.add_parser("features", help="moteur C sur tout le flux → data/features/<variante>/<config>")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--motif-formats", default="", help="formats du graphe de motifs, ex. ACH,Wire")
    s.add_argument("--ring-bits", type=int, default=0)
    s.add_argument("--hindsight", action="store_true", help="calcule aussi les features J+1 (H.npy)")
    s.set_defaults(func=_cmd_features)

    s = sub.add_parser("rules", help="baseline B0 : règles expertes, seuils fixés sur la validation")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.set_defaults(func=_cmd_rules)

    s = sub.add_parser("ablate", help="ablations par famille de features (validation)")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--only", default="", help="sous-ensemble d'ablations, séparées par des virgules")
    s.add_argument("--seeds", default="42")
    s.add_argument("--neg-rate", type=float, default=0.25)
    s.set_defaults(func=_cmd_ablate)

    s = sub.add_parser("train", help="modèle livré (forêt + seuil de validation) dans artifacts/models/<nom>")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--name", default="strict")
    s.add_argument("--families", default="")
    s.add_argument("--seeds", default="42,43,44,45,46")
    s.add_argument("--neg-rate", type=float, default=0.25)
    s.add_argument("--hindsight", action="store_true", help="modèle revue J+1 (features rétrospectives)")
    s.add_argument("--stage2", action="store_true", help="modèle d'étape 2 (propagation des scores)")
    s.add_argument("--train-days", default="", help="sous-période du train en jours, ex. 0,3")
    s.set_defaults(func=_cmd_train)

    s = sub.add_parser("stage2-features", help="features avec propagation (validation croisée temporelle)")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.set_defaults(func=_cmd_stage2_features)

    s = sub.add_parser("evaluate", help="évalue un modèle figé (seuil de validation) sur des périodes")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--model", default="strict")
    s.add_argument("--splits", default="val", help="ex. test,tail,test_lit (le test n'est lu qu'ici)")
    s.add_argument("--cascade", action="store_true", help="applique le pré-filtrage comme en ligne")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("transfer", help="modèle figé évalué sur une autre variante entière (results/transfer_*)")
    s.add_argument("--variant", required=True, help="ex. HI-Medium, LI-Small")
    s.add_argument("--model", default="strict")
    s.add_argument("--no-cascade", action="store_true", help="forêt complète sur chaque transaction (lent)")
    s.add_argument("--threads", type=int, default=0, help="threads de notation (0 = min(8, cœurs))")
    s.set_defaults(func=_cmd_transfer)

    s = sub.add_parser("bench", help="frugalité : débit, latences, RAM, tailles, énergie (results/bench_*.json)")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--model", default="strict")
    s.add_argument("--watts-idle", type=float, default=None, help="puissance au repos lue au wattmètre (W)")
    s.add_argument("--watts-load", type=float, default=None, help="puissance en charge lue au wattmètre (W)")
    s.add_argument("--tag", default="", help="suffixe du fichier de résultats (ex. cpu7)")
    s.add_argument("--no-cascade", action="store_true", help="désactive le pré-filtrage (comparaison)")
    s.set_defaults(func=_cmd_bench)

    s = sub.add_parser("demo", help="rejeu en flux, alertes en direct, cas et rapports (out/demo)")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--model", default="strict")
    s.add_argument("--from", dest="start", default="2022-09-09")
    s.add_argument("--days", type=int, default=1)
    s.add_argument("--reports", type=int, default=10)
    s.add_argument("--llm", action="store_true", help="reformulation de l'analyse par un LLM local (sous garde)")
    s.set_defaults(func=_cmd_demo)

    s = sub.add_parser("ask", help="question à l'assistant sur un cas de la démo")
    s.add_argument("case")
    s.add_argument("question")
    s.add_argument("--llm", action="store_true")
    s.set_defaults(func=_cmd_ask)

    s = sub.add_parser("explain-eval", help="fidélité et couverture des explications sur une période")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--model", default="strict")
    s.add_argument("--split", default="val")
    s.set_defaults(func=_cmd_explain_eval)

    s = sub.add_parser("export-lgbm", help="régénère lgbm.txt (TreeSHAP) des modèles livrés, forêt vérifiée identique")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--models", default="strict,j1")
    s.set_defaults(func=_cmd_export_lgbm)

    s = sub.add_parser("cascade", help="mini-modèle de pré-filtrage (cascade), seuil fixé sur la validation")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--model", default="strict")
    s.set_defaults(func=_cmd_cascade)

    s = sub.add_parser("llm-bench", help="compare les LLM locaux sur les dossiers de la démo (garde comprise)")
    s.add_argument("--n", type=int, default=10)
    s.add_argument("--threads", type=int, default=4)
    s.set_defaults(func=_cmd_llm_bench)

    s = sub.add_parser("pareto", help="taille de forêt contre qualité (validation) et coût mesuré")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.add_argument("--model", default="strict")
    s.set_defaults(func=_cmd_pareto)

    s = sub.add_parser("summarize", help="tableaux Markdown issus de results/*.json")
    s.set_defaults(func=_cmd_summarize)

    s = sub.add_parser("typology", help="classifieur de typologie (illicites du train, validation)")
    s.add_argument("--variant", default=DEFAULT_VARIANT)
    s.set_defaults(func=_cmd_typology)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
