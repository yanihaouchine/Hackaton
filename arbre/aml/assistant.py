"""Assistant d'analyse : répond aux questions de l'analyste à partir du seul dossier du cas.

Les questions fréquentes reçoivent une réponse par template (sans LLM, instantanée) ; les autres passent par
le LLM sous la même garde que le rapport (jetons, aucun chiffre écrit par le modèle). Sans réponse fondée
dans le dossier, l'assistant le dit.
"""

from __future__ import annotations

import re

from aml.llm import LLM, guard, slotify

INTENTS = [
    ("pourquoi", r"pourquoi|raison|motif|justif|explique"),
    ("comptes", r"compte|qui |titulaire|banque|pays|entit"),
    ("montants", r"montant|combien|total|somme|argent|usd"),
    ("chronologie", r"quand|date|chronolog|période|periode|délai|delai"),
    ("typologie", r"typolog|schéma|schema|type de|mécanisme|mecanisme|circuit|cycle"),
    ("suite", r"faire|suite|recommand|action|prochaine|étape|etape|déclar|declar|dois"),
    ("fiabilite", r"fiab|confiance|score|certain|probab|faux positif"),
]


def intent(question: str) -> str | None:
    q = question.lower()
    for name, pat in INTENTS:
        if re.search(pat, q):
            return name
    return None


def answer_template(f: dict, kind: str) -> str:
    if kind == "pourquoi":
        rs = "\n".join(f"- {r['family']} : {r['text']}" for r in f["reasons"])
        inds = "\n".join(f"- {i['text']}" for i in f["indicators"]) or "- aucun indicateur à règle fixe"
        return f"L'alerte repose sur ces éléments (par contribution décroissante) :\n{rs}\nIndicateurs :\n{inds}"
    if kind == "comptes":
        return "\n".join(f"- `{a['id']}` ({a['role']}) : {a['bank']}, {a['country']}, titulaire {a['entity']}, "
                         f"{a['n_ops']} opération(s) dans le dossier" for a in f["accounts"])  # fmt: skip
    if kind == "montants":
        top = sorted(f["operations"], key=lambda o: -o["usd"])[:5]
        ops = "\n".join(f"- {o['date']} : {o['amount_fmt']} {o['currency']} ({o['usd_fmt']} USD), {o['format']}"
                        for o in top)  # fmt: skip
        head = f"Total du dossier : {f['totals']['usd_fmt']} sur {f['totals']['n_ops']} opérations."
        return f"{head} Les plus élevées :\n{ops}"
    if kind == "chronologie":
        ops = "\n".join(f"- {o['date']} : `{o['src']}` → `{o['dst']}`, {o['amount_fmt']} {o['currency']}"
                        for o in f["operations"][:15])  # fmt: skip
        return f"Opérations du {f['period']['start']} au {f['period']['end']} :\n{ops}"
    if kind == "typologie":
        ev = "\n".join(f"- {line}" for line in f["evidence_lines"]) or "- pas de structure reconstituée"
        return f"Typologie probable : {f['typology']['label']} (confiance {f['typology']['proba_pct']}).\n{ev}"
    if kind == "suite":
        return (
            "Suggestions (à apprécier par l'analyste) : vérifier la cohérence avec la connaissance client "
            "(activité déclarée, justificatifs des opérations) ; examiner les contreparties récurrentes ; "
            "si le soupçon persiste, préparer la déclaration via ERMES sans en informer le client."
        )
    if kind == "fiabilite":
        return (
            f"Score maximal du dossier : {f['max_score_pct']} (seuil d'alerte {f['threshold_pct']}). "
            f"Probabilité calibrée qu'au moins une opération soit illicite : {f['priority_pct']}. "
            "Le modèle a été évalué sur des périodes postérieures à son apprentissage ; la décision reste humaine."
        )
    return ""


def ask(f: dict, question: str, llm: LLM | None = None) -> dict:
    kind = intent(question)
    if kind:
        return {"source": "template", "intent": kind, "text": answer_template(f, kind)}
    if llm is None or not llm.available:
        return {"source": "template", "intent": None,
                "text": "Je ne trouve pas de réponse fondée dans le dossier ; reformulez (raisons, comptes, "
                        "montants, chronologie, typologie, suite à donner, fiabilité)."}  # fmt: skip
    lines = [f["analysis_text"], *[r["text"] for r in f["reasons"]], *f["evidence_lines"],
             *[f"compte `{a['id']}` : {a['role']}, {a['bank']}, {a['country']}" for a in f["accounts"]]]  # fmt: skip
    masked, slots = slotify(lines)
    system = ("Tu assistes un analyste LCB-FT. Tu réponds en français, brièvement, UNIQUEMENT à partir du dossier "
              "fourni. Tu n'écris aucun chiffre : recopie les jetons (⟦TYPE_n⟧) tels quels, en respectant "
              "leur type. Si la réponse n'est pas dans le dossier, réponds exactement : "
              "« Absent du dossier. »")  # fmt: skip
    user = "Dossier :\n- " + "\n- ".join(masked) + f"\n\nQuestion : {question}"
    raw = llm.chat(system, user, max_tokens=250)
    if "Absent du dossier" in raw:
        return {"source": f"llm:{llm.name}", "intent": None, "text": "Absent du dossier."}
    ok, text = guard(raw, slots)
    if not ok:
        return {"source": "template", "intent": None,
                "text": f"Réponse du LLM rejetée par la garde ({text}) : consultez le rapport."}  # fmt: skip
    return {"source": f"llm:{llm.name}", "intent": None, "text": text}
