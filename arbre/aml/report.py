"""Rapport de type TRACFIN : dossier de faits (JSON) → Markdown français via un template Jinja2.

Le texte d'analyse est produit par un template ; un LLM local peut le reformuler (aml/llm.py) sous une
garde stricte, mais tous les nombres, dates et identifiants viennent du dossier.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from aml.config import TYPOLOGY_FR
from aml.dossier import account_facts, evidence, tx_record
from aml.explain import _num, contributions, fmt_dur, fmt_pct, fmt_usd, reasons

TEMPLATES = Path(__file__).with_name("templates")

# Indicateurs à seuils fixes (lisibles par un analyste) : feature, sens, seuil, libellé
INDICATORS = [
    ("g_s_outdeg_l", ">=", 5, "éventail sortant : l'émetteur a payé {v} bénéficiaires distincts en 96 h"),
    ("g_r_indeg_l", ">=", 5, "éventail entrant : le destinataire a reçu de {v} émetteurs distincts en 96 h"),
    ("f_best_err", "<=", 0.01, "fonds de passage : montant identique à {p} près à une somme reçue juste avant"),
    ("f_chain_depth", ">=", 2, "chaîne de passage : {v} transferts successifs de montants quasi identiques"),
    ("m_cyc_n", ">=", 1, "circuit fermé : les fonds reviennent au compte d'origine"),
    ("m_sg", ">=", 2, "dispersion puis collecte : {v} intermédiaires entre une même source et ce destinataire"),
    ("g_s_indeg_l", ">=", 3, "collecte puis dispersion : l'émetteur redistribue des fonds reçus de {v} émetteurs"),
]


def indicators(x_row: np.ndarray, names: list[str]) -> list[dict]:
    col = {n: i for i, n in enumerate(names)}
    out = []
    for feat, op, thr, text in INDICATORS:
        v = float(x_row[col[feat]])
        if (op == ">=" and v >= thr) or (op == "<=" and v <= thr):
            out.append({"feature": feat, "value": v, "text": text.format(v=_num(v), p=fmt_pct(v, 2))})
    return out


def _percentile_txt(p: float) -> str:
    if p != p:  # NaN
        return ""
    if p >= 0.9:
        return f"valeur plus élevée que {fmt_pct(p, 1)} des transactions d'apprentissage"
    if p <= 0.1:
        return f"valeur plus faible que {fmt_pct(1 - p, 1)} des transactions d'apprentissage"
    return ""


class Reporter:
    def __init__(self, stream, bundle, history, acc_table: dict[int, dict], typology=None, mode: str = "strict"):
        self.s = stream
        self.b = bundle
        self.h = history
        self.acc = acc_table
        self.typ = typology
        self.mode = mode
        self.t0 = datetime.fromisoformat(stream.meta["t0"])
        self.countries = stream.meta["countries"]

    def _date(self, t: int) -> str:
        return (self.t0 + timedelta(minutes=int(t))).strftime("%d/%m/%Y %H:%M")

    def facts(self, case, x_alerts: np.ndarray, p_alerts: np.ndarray, rank: int = 1) -> dict:
        """x_alerts / p_alerts : features moteur et probabilités brutes des alertes du cas (même ordre)."""
        s, names = self.s, self.b.forest.engine_features
        alerts = list(case.alerts)
        top = int(np.argmax(p_alerts))
        contrib, method = contributions(self.b, x_alerts[top : top + 1])
        rs = reasons(self.b, x_alerts[top], contrib[0], self.countries)
        for r in rs:
            r["percentile_txt"] = _percentile_txt(r["percentile"])
        inds, seen = [], set()
        for k in range(len(alerts)):
            for ind in indicators(x_alerts[k], names):
                if ind["feature"] not in seen:
                    seen.add(ind["feature"])
                    inds.append(ind)
        typ_code, typ_p = "OTHER", 0.0
        if self.typ is not None:
            _, pr = self.typ.predict(x_alerts)
            w = np.asarray(p_alerts, dtype=np.float64)[:, None]
            agg = (pr * w).sum(0) / max(w.sum(), 1e-12)
            j = int(agg.argmax())
            typ_code, typ_p = self.typ.classes[j], float(agg[j])
        key = {a: self.acc[a]["key"] for a in case.accounts}
        cal = self.b.calibrate(p_alerts)
        ops = []
        for j, p in zip(alerts, cal, strict=True):
            o = tx_record(s, j, self.t0, key)
            o.update(score_pct=fmt_pct(float(p)), amount_fmt=_num(o["amount"], 2), usd_fmt=_num(o["usd"], 2))
            ops.append(o)
        ops.sort(key=lambda o: o["index"])
        n_ops = {a: 0 for a in case.accounts}
        role = {a: set() for a in case.accounts}
        for j in alerts:
            n_ops[int(s.src[j])] += 1
            n_ops[int(s.dst[j])] += 1
            role[int(s.src[j])].add("émetteur")
            role[int(s.dst[j])].add("destinataire")
        accounts = []
        for a in case.accounts:
            af = account_facts(self.acc, a, self.countries, s)
            r = role[a]
            af.update(role="intermédiaire" if len(r) == 2 else next(iter(r)), n_ops=n_ops[a])
            accounts.append(af)
        ev = evidence(self.h, alerts[top])
        f = {
            "case_id": case.case_id,
            "generated": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "priority": case.priority,
            "priority_pct": "> 99,9 %" if case.priority > 0.999 else fmt_pct(case.priority),
            "priority_rank": rank,
            "typology": {
                "code": typ_code,
                "label": TYPOLOGY_FR.get(typ_code, typ_code),
                "proba": typ_p,
                "proba_pct": fmt_pct(typ_p, 0),
            },  # fmt: skip
            "period": {"start": self._date(case.t_first), "end": self._date(case.t_last)},
            "totals": {
                "n_ops": len(alerts),
                "n_accounts": len(case.accounts),
                "usd": case.total_usd,
                "usd_fmt": fmt_usd(case.total_usd),
            },  # fmt: skip
            "accounts": accounts,
            "operations": ops[:50],
            "reasons": rs,
            "indicators": inds,
            "evidence": ev,
            "evidence_lines": self._evidence_lines(ev, key, alerts[top]),
            "max_score_pct": "> 99,9 %" if float(cal.max()) > 0.999 else fmt_pct(float(cal.max())),
            "threshold_pct": fmt_pct(float(self.b.calibrate(np.array([self.b.threshold]))[0])),
            "model": {
                "name": self.b.name,
                "method": method,
                "mode": self.mode,
                "mode_label": "temps réel (score à l'arrivée)"
                if self.mode == "strict"
                else "revue J+1 (données jusqu'à la décision)",
            },  # fmt: skip
            "narrative_note": "le texte d'analyse est produit par un modèle de phrases (sans LLM).",
        }
        f["analysis_text"] = narrative(f)
        return f

    def _evidence_lines(self, ev: dict, key: dict, i: int) -> list[str]:
        s, out = self.s, []

        def name(a):
            return f"`{self.acc[int(a)]['key']}`"

        if ev["cycle_tx"]:
            c = ev["cycle_tx"]
            path = " → ".join([name(s.src[c[0]])] + [name(s.dst[j]) for j in c])
            out.append(f"Circuit fermé de {len(c)} virements : {path}, du {self._date(s.t[c[0]])} au "
                       f"{self._date(s.t[c[-1]])} ({fmt_usd(float(s.usd_paid[c[0]]))} au départ, "
                       f"{fmt_usd(float(s.usd_paid[c[-1]]))} au retour).")  # fmt: skip
        tr = ev["pass_through_trail"]
        if tr:
            chain = [*tr, i]
            path = " → ".join([name(s.src[chain[0]])] + [name(s.dst[j]) for j in chain])
            lag = int(s.t[i] - s.t[tr[-1]])
            out.append(f"Chaîne de passage de {len(chain)} transferts de montants quasi identiques : {path} "
                       f"({fmt_usd(float(s.usd_paid[chain[0]]))} puis {fmt_usd(float(s.usd_paid[i]))} ; "
                       f"dernier relais en {fmt_dur(lag)}).")  # fmt: skip
        if ev["fan_out_distinct"] >= 3:
            out.append(f"Le compte {name(s.src[i])} a payé {ev['fan_out_distinct']} bénéficiaires distincts en 96 h "
                       f"({len(ev['fan_out_tx'])} virements ACH).")  # fmt: skip
        if ev["fan_in_distinct"] >= 3:
            out.append(f"Le compte {name(s.dst[i])} a reçu de {ev['fan_in_distinct']} émetteurs distincts en 96 h "
                       f"({len(ev['fan_in_tx'])} virements ACH).")  # fmt: skip
        if ev["scatter_gather_tx"]:
            w = len(set(int(s.dst[j]) for j in ev["scatter_gather_tx"]))
            out.append(f"{w} intermédiaire(s) relient une même source au compte {name(s.dst[i])} "
                       f"(dispersion puis collecte).")  # fmt: skip
        return out

    def render(self, facts: dict) -> str:
        from jinja2 import Environment, FileSystemLoader

        env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=False, trim_blocks=True)
        return env.get_template("rapport.md.j2").render(f=facts)


def narrative(f: dict) -> str:
    """Analyse rédigée par template : toutes les valeurs proviennent du dossier."""
    t = f["typology"]
    parts = [
        f"Du {f['period']['start']} au {f['period']['end']}, {f['totals']['n_accounts']} comptes ont réalisé "
        f"{f['totals']['n_ops']} opérations signalées pour un total de {f['totals']['usd_fmt']}.",
        f"Le schéma observé est rapproché de la typologie « {t['label']} » (confiance {t['proba_pct']}).",
    ]
    main = [r for r in f["reasons"] if not r.get("context")] or f["reasons"]
    if main:
        top = " ; ".join(r["text"] for r in main[:3])
        parts.append(f"Les éléments les plus déterminants sont : {top}.")
    if f["evidence_lines"]:
        parts.append(f["evidence_lines"][0])
    parts.append("Ces éléments, inhabituels au regard de l'activité passée des comptes, justifient un examen "
                 "approfondi et, le cas échéant, une déclaration de soupçon.")  # fmt: skip
    return " ".join(parts)
