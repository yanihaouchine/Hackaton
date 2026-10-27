"""Raisons lisibles d'une alerte : contributions TreeSHAP (lightgbm) ou Saabas (repli numpy), traduites en
phrases françaises avec la valeur observée et son rang parmi les transactions d'apprentissage."""

from __future__ import annotations

import math
import re

import numpy as np

from aml.config import CURRENCIES, ENTITY_TYPES, FORMATS

NO_MATCH = 9.0
SCALE_FR = {"s": "6 h", "m": "24 h", "l": "96 h"}
ROLE_FR = {"s": "l'émetteur", "r": "le destinataire"}
ROLE_DE = {"s": "de l'émetteur", "r": "du destinataire"}  # contraction après « de »
ETYPE_FR = {"Corporation": "société", "Partnership": "société de personnes",
            "Sole Proprietorship": "entreprise individuelle",
            "Individual": "particulier", "Country": "entité publique", "Direct": "compte direct"}  # fmt: skip


def _num(x: float, nd: int = 0) -> str:
    s = f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")
    return s


def fmt_usd(v: float) -> str:
    return f"{_num(v, 2)} USD"


def fmt_dur(minutes: float) -> str:
    if minutes < 0:
        return "jamais"
    if minutes < 60:
        return f"{_num(minutes)} min"
    if minutes < 1440:
        return f"{_num(minutes / 60, 1)} h"
    return f"{_num(minutes / 1440, 1)} j"


def fmt_pct(x: float, nd: int = 1) -> str:
    return f"{_num(100 * x, nd)} %"


def _match_text(v: float) -> str:
    """Appariement de montant : la formulation suit la force du signal (pas de « passage » pour 16 %)."""
    if v >= NO_MATCH:
        return "aucune entrée récente comparable chez l'émetteur"
    if v <= 0.02:
        return f"le montant reproduit à {fmt_pct(v, 2)} près une somme reçue par l'émetteur (fonds de passage)"
    if v <= 0.10:
        return f"le montant est proche à {fmt_pct(v, 1)} près d'une somme reçue par l'émetteur"
    return f"la somme reçue la plus proche chez l'émetteur s'écarte de {fmt_pct(v, 0)} de ce montant"


def describe(name: str, v: float, countries: list[str] | None = None) -> str:
    """Phrase française décrivant la valeur d'une feature du moteur."""
    countries = countries or []
    m = re.match(r"^([sr])_(.*)$", name)
    if m:
        who, f = ROLE_FR[m.group(1)], m.group(2)
        de_who = ROLE_DE[m.group(1)]
        mm = re.match(r"^ew([ca])_(out|in)_([sml])$", f)
        if mm:
            sens = "envoyés par" if mm.group(2) == "out" else "reçus par"
            if mm.group(1) == "c":
                return f"≈{_num(v, 1)} virements {sens} {who} sur {SCALE_FR[mm.group(3)]} (moyenne glissante)"
            return f"≈{fmt_usd(math.expm1(v))} {sens} {who} sur {SCALE_FR[mm.group(3)]} (moyenne glissante)"
        table = {
            "age": lambda: f"{who} est un compte nouveau" if v < 0 else f"{who} est actif depuis {fmt_dur(v)}",
            "dt_out": lambda: f"dernier envoi {de_who} : {'aucun' if v < 0 else 'il y a ' + fmt_dur(v)}",
            "dt_in": lambda: f"dernière réception {de_who} : {'aucune' if v < 0 else 'il y a ' + fmt_dur(v)}",
            "n_out": lambda: f"{_num(v)} envois antérieurs {de_who}",
            "n_in": lambda: f"{_num(v)} réceptions antérieures {de_who}",
            "z_out": lambda: f"montant à {_num(v, 1)} écarts-types des envois habituels {de_who}",
            "z_in": lambda: f"montant à {_num(v, 1)} écarts-types des réceptions habituelles {de_who}",
            "ach_out": lambda: "aucun envoi antérieur" if v < 0 else f"{fmt_pct(v, 0)} des envois {de_who} en ACH",
            "ach_in": lambda: (
                "aucune réception antérieure" if v < 0 else f"{fmt_pct(v, 0)} des réceptions {de_who} en ACH"
            ),
            "dist_out": lambda: f"{who} a payé {_num(v)} bénéficiaires distincts depuis le début du flux",
            "dist_in": lambda: f"{who} a reçu de {_num(v)} émetteurs distincts depuis le début du flux",
            "flow": lambda: f"sorties / entrées {de_who} sur 96 h : ×{_num(math.exp(v), 2)}",
            "n_self": lambda: f"{_num(v)} réinvestissements internes {de_who}",
        }
        if f in table:
            return table[f]()
    fmt_code = FORMATS[int(v)] if name == "tx_fmt" and 0 <= v < len(FORMATS) else "?"
    cur_code = CURRENCIES[int(v)] if name == "tx_cur" and 0 <= v < len(CURRENCIES) else "?"
    country = countries[int(v)] if name.endswith("_country") and 0 <= v < len(countries) else "?"
    et = ENTITY_TYPES[int(v)] if name.endswith("_etype") and 0 <= v < len(ENTITY_TYPES) else "autre"
    table = {
        "tx_log_usd": lambda: f"montant de {fmt_usd(math.expm1(v))}",
        "tx_fmt": lambda: f"moyen de paiement : {fmt_code}",
        "tx_is_ach": lambda: "virement ACH" if v > 0 else "paiement hors ACH",
        "tx_cur": lambda: f"devise : {cur_code}",
        "tx_xcur": lambda: "opération avec change de devise" if v > 0 else "pas de change de devise",
        "tx_self": lambda: "virement vers soi-même" if v > 0 else "virement vers un tiers",
        "tx_same_bank": lambda: "émetteur et destinataire dans la même banque" if v > 0 else "banques différentes",
        "tx_hour": lambda: f"heure de l'opération : {int(v)} h",
        "tx_round": lambda: f"montant rond ({int(v)} zéros finaux en centimes)",
        "kyc_s_etype": lambda: f"émetteur : {ETYPE_FR.get(et, et)}",
        "kyc_r_etype": lambda: f"destinataire : {ETYPE_FR.get(et, et)}",
        "kyc_s_nacc": lambda: f"l'entité émettrice détient {_num(v)} comptes",
        "kyc_r_nacc": lambda: f"l'entité destinataire détient {_num(v)} comptes",
        "kyc_same_entity": lambda: "même titulaire des deux côtés" if v > 0 else "titulaires différents",
        "kyc_s_country": lambda: f"banque émettrice : {country}",
        "kyc_r_country": lambda: f"banque destinataire : {country}",
        "kyc_cross_border": lambda: "transfert transfrontalier" if v > 0 else "transfert domestique",
        "p_cnt": lambda: (
            "première relation entre ces deux comptes"
            if v == 0
            else f"{_num(v)} virements antérieurs entre ces deux comptes"
        ),
        "p_dt": lambda: (
            "aucun virement antérieur entre ces comptes"
            if v < 0
            else f"dernier virement entre ces comptes il y a {fmt_dur(v)}"
        ),
        "p_rev_cnt": lambda: f"{_num(v)} virements antérieurs en sens inverse (retour de fonds)",
        "p_rev_dt": lambda: (
            "aucun virement en sens inverse" if v < 0 else f"dernier virement en sens inverse il y a {fmt_dur(v)}"
        ),
        "g_s_outdeg_m": lambda: f"l'émetteur a payé {_num(v)} bénéficiaires distincts en 24 h (ACH)",
        "g_s_outdeg_l": lambda: f"l'émetteur a payé {_num(v)} bénéficiaires distincts en 96 h (ACH)",
        "g_s_outcnt_l": lambda: f"l'émetteur a émis {_num(v)} virements ACH en 96 h",
        "g_s_indeg_l": lambda: f"l'émetteur a reçu de {_num(v)} émetteurs distincts en 96 h (ACH)",
        "g_r_indeg_m": lambda: f"le destinataire a reçu de {_num(v)} émetteurs distincts en 24 h (ACH)",
        "g_r_indeg_l": lambda: f"le destinataire a reçu de {_num(v)} émetteurs distincts en 96 h (ACH)",
        "g_r_incnt_l": lambda: f"le destinataire a reçu {_num(v)} virements ACH en 96 h",
        "g_r_outdeg_l": lambda: f"le destinataire a payé {_num(v)} bénéficiaires distincts en 96 h (ACH)",
        "f_best_err": lambda: _match_text(v),
        "f_best_lag": lambda: "—" if v < 0 else f"la somme comparable avait été reçue {fmt_dur(v)} plus tôt",
        "f_n_match2": lambda: f"{_num(v)} entrées récentes de l'émetteur à ±2 % de ce montant",
        "f_n_match10": lambda: f"{_num(v)} entrées récentes de l'émetteur à ±10 % de ce montant",
        "f_chain_depth": lambda: f"maillon n°{_num(v + 1)} d'une chaîne de transferts de montants quasi identiques",
        "f_chain_age": lambda: "—" if v < 0 else f"chaîne de transferts commencée il y a {fmt_dur(v)}",
        "f_in_sum_l": lambda: f"{fmt_usd(math.expm1(v))} reçus par l'émetteur en 96 h (ACH)",
        "f_out_sum_l": lambda: f"{fmt_usd(math.expm1(v))} déjà émis par l'émetteur en 96 h (ACH)",
        "f_split": lambda: f"sorties cumulées / entrées de l'émetteur sur 96 h : ×{_num(math.exp(v), 2)}",
        "f_r_ret_err": lambda: (
            "—"
            if v >= NO_MATCH
            else f"le destinataire avait lui-même envoyé un montant identique à {fmt_pct(v, 2)} près (retour)"
        ),
        "m_cyc_n": lambda: f"{_num(v)} circuit(s) de fonds refermé(s) par ce virement",
        "m_cyc_minlen": lambda: f"circuit de {_num(v)} virements revenant au compte d'origine",
        "m_cyc_drift": lambda: "—" if v >= NO_MATCH else f"écart de montant sur le circuit : {fmt_pct(v)}",
        "m_cyc4_n": lambda: f"{_num(v)} circuit(s) court(s) (≤ 4 virements) refermé(s) par ce virement",
        "m_sg": lambda: (
            f"{_num(v)} intermédiaires relient une même source à ce destinataire (dispersion puis collecte)"
        ),
        "m_2hop": lambda: f"{_num(v)} intermédiaires communs entre l'émetteur et le destinataire",
        "m_bip": lambda: f"{_num(v)} liens croisés avec d'autres bénéficiaires de l'émetteur (réseau biparti)",
    }
    fn = table.get(name)
    return fn() if fn else f"{name} = {_num(v, 3)}"


FAMILY_FR = {"tx": "Opération", "kyc": "Connaissance client", "s": "Comportement de l'émetteur",
             "r": "Comportement du destinataire", "p": "Relation", "g": "Structure (éventails)",
             "f": "Conservation des flux", "m": "Motifs (circuits, dispersion)"}  # fmt: skip


def family(name: str) -> str:
    return FAMILY_FR.get(name.split("_", 1)[0], "Autre")


CATEGORICAL = {"tx_fmt", "tx_is_ach", "tx_cur", "tx_xcur", "tx_self", "tx_same_bank", "tx_hour", "tx_round",
               "kyc_s_etype", "kyc_r_etype", "kyc_s_country", "kyc_r_country", "kyc_cross_border",
               "kyc_same_entity"}  # fmt: skip
# valeurs sentinelles (9 = aucun appariement, -1 = jamais) : un rang parmi l'apprentissage n'aurait pas de sens
NO_PERCENTILE = {"f_best_err", "f_r_ret_err", "m_cyc_drift", "f_best_lag", "f_chain_age", "p_dt", "p_rev_dt",
                 "s_age", "r_age", "s_dt_out", "s_dt_in", "r_dt_out", "r_dt_in"}  # fmt: skip


def is_context(name: str) -> bool:
    """Contexte de l'opération (format, montant, KYC…) par opposition au comportement et à la structure."""
    return name.startswith(("tx_", "kyc_"))


def _group(name: str) -> str:
    """Regroupe les variantes d'une même information pour ne pas répéter une raison."""
    if name in ("tx_fmt", "tx_is_ach"):
        return "format"
    if name in ("p_cnt", "p_dt"):
        return "relation"
    if name in ("p_rev_cnt", "p_rev_dt"):
        return "relation_inverse"
    if name.startswith("kyc_") and name.endswith("_country"):
        return "pays"
    return re.sub(r"_(s|m|l)$", "", name) if re.search(r"_ew[ca]_", name) or name.startswith("g_") else name


def contributions(bundle, x: np.ndarray) -> tuple[np.ndarray, str]:
    """Contributions par feature du moteur (n, F+1) ; méthode utilisée en second retour."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    nf = len(bundle.forest.engine_features)
    try:
        import lightgbm as lgb

        if not bundle.lgbm_path().exists():  # non livré ou non régénéré : repli Saabas
            raise FileNotFoundError(bundle.lgbm_path())
        b = lgb.Booster(model_file=str(bundle.lgbm_path()))
        c = b.predict(x[:, bundle.cols], pred_contrib=True)
        out = np.zeros((len(x), nf + 1))
        out[:, bundle.cols] = c[:, :-1]
        out[:, nf] = c[:, -1]
        return out, "TreeSHAP"
    except (ImportError, OSError):
        return bundle.forest.saabas(x), "Saabas"


def reasons(bundle, x_row: np.ndarray, contrib_row: np.ndarray, countries: list[str], k: int = 5,
            k_context: int = 3) -> list[dict]:  # fmt: skip
    """Raisons à contribution positive, sans doublon : d'abord le comportement et la structure (k au plus),
    puis le contexte de l'opération (k_context au plus), marqué comme tel."""
    names = bundle.forest.engine_features
    order = np.argsort(-contrib_row[: len(names)])
    main, ctx, seen = [], [], set()
    for j in order:
        c = float(contrib_row[j])
        if c <= 0:
            break
        n = names[j]
        g = _group(n)
        if g in seen:
            continue
        seen.add(g)
        v = float(x_row[j])
        pct = float("nan") if n in CATEGORICAL or n in NO_PERCENTILE else bundle.percentile(n, v)
        r = {"feature": n, "family": family(n), "value": v, "contribution": c, "percentile": pct,
             "text": describe(n, v, countries), "context": is_context(n)}  # fmt: skip
        (ctx if r["context"] else main).append(r)
    return main[:k] + ctx[:k_context]


def faithfulness(bundle, x: np.ndarray, contrib: np.ndarray, k: int = 3) -> dict:
    """Part des alertes qui repassent sous le seuil quand on remplace leurs k premières raisons par la médiane
    du train : mesure que les raisons affichées portent réellement la décision."""
    x = np.array(x, dtype=np.float32, copy=True)
    names = bundle.forest.engine_features
    med = np.array([bundle.quantiles[n][4] for n in names], dtype=np.float32)  # quantile 0,5
    top = np.argsort(-contrib[:, : len(names)], axis=1)[:, :k]
    before = bundle.forest.predict(x)
    for i in range(len(x)):
        x[i, top[i]] = med[top[i]]
    after = bundle.forest.predict(x)
    thr = bundle.threshold
    alerted = before >= thr
    return {"k": k, "n_alerts": int(alerted.sum()),
            "share_below_threshold_after": float((after[alerted] < thr).mean()) if alerted.any() else None,
            "median_score_drop": float(np.median(before[alerted] - after[alerted])) if alerted.any()
            else None}  # fmt: skip
