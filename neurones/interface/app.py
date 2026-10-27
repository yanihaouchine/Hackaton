"""Interface de suivi des alertes (démo) : file des cas, raisons, graphe des comptes, rapport TRACFIN, décision.
Lancement (depuis la racine du dépôt) : bash neurones/interface/run.sh
Streamlit ne sert qu'à la démo : il lit les dossiers d'alerte écrits par la détection, il n'en fait pas partie.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # neurones/

import pandas as pd
import streamlit as st

from interface import logic
from rapport.dossier import main_account
from rapport.render import reason_line, render

st.set_page_config(page_title="Suivi des alertes LCB-FT", layout="wide")
st.title("Suivi des alertes LCB-FT")
st.caption("Données synthétiques IBM AML. Chaque alerte est un brouillon : l'analyste décide.")


@st.cache_resource
def llm_client():
    """Client du LLM local (Ollama ou llama-server, AML_LLM_URL / AML_LLM_MODEL), ou None s'il est absent."""
    from rapport.llm import LlmUnavailable, OpenAICompatible

    try:
        return OpenAICompatible()
    except LlmUnavailable:
        return None


days = logic.days()
if not days:
    st.error("Aucune alerte. Lancer d'abord : `cd neurones && ../.venv/bin/python -m rapport.make_alerts`")
    st.stop()

day = st.sidebar.selectbox("Journée", days)
cases = logic.cases(day)
done = logic.decisions(day)
typologies = sorted({c["typology"] for c in cases})
chosen_typ = st.sidebar.multiselect("Typologies", typologies, default=typologies)
status = st.sidebar.radio("Statut", ["Tous", "À traiter", "Traités"])

rows = []
for c in cases:
    d = done.get(c["case_id"])
    if c["typology"] not in chosen_typ or (status == "À traiter" and d) or (status == "Traités" and not d):
        continue
    rows.append({"Rang": c["rank"], "Cas": c["case_id"], "Compte principal": c["main_account"],
                 "Typologie": c["typology"], "Score (priorité)": c["priority_pct"], "Alertes": c["n_alerts"],
                 "Montant": c["usd_fmt"], "Décision": logic.LABELS[d["decision"]] if d else "à traiter"})  # fmt: skip

st.sidebar.metric("Cas de la journée", len(cases))
st.sidebar.metric("Décisions prises", len(done))

st.subheader("File des alertes, triées par score")
if not rows:
    st.info("Aucun cas ne correspond aux filtres.")
    st.stop()
table = pd.DataFrame(rows)
picked = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="single-row", key="file")  # fmt: skip
sel = picked.selection.rows if picked and picked.selection else []
case_id = table.iloc[sel[0]]["Cas"] if sel else table.iloc[0]["Cas"]
if not sel:
    st.caption("Cliquer sur une ligne pour ouvrir un cas ; le plus prioritaire est affiché par défaut.")

f = logic.dossier(day, case_id)
main = main_account(f)
st.subheader(f"Cas {case_id} : {f['typology']['label']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Priorité", f["priority_pct"])
c2.metric("Opérations", f["totals"]["n_ops"])
c3.metric("Comptes", f["totals"]["n_accounts"])
c4.metric("Montant", f["totals"]["usd_fmt"])
st.write(f"Compte principal : `{main['id']}` ({main['entity']}, {main['bank']}). "
         f"Du {f['period']['start']} au {f['period']['end']}.")  # fmt: skip

t_reasons, t_graph, t_report = st.tabs(["Raisons de l'alerte", "Graphe des comptes", "Rapport TRACFIN"])
with t_reasons:
    st.markdown(f"**Pourquoi ce cas est suspect** (méthode : {f['model']['method']}) :")
    for r in f["reasons"]:
        st.markdown(f"- {reason_line(r)}")
    extra = [i["text"] for i in f.get("indicators", [])] + f.get("evidence_lines", [])
    if extra:
        st.markdown("**Indicateurs du réseau de comptes :**")
        for e in extra:
            st.markdown(f"- {e}")
with t_graph:
    st.caption("Une flèche par opération signalée (date, montant). En orange : le compte principal.")
    st.graphviz_chart(logic.case_dot(f, main["id"]), width="stretch")
with t_report:
    writer = None
    if st.toggle("Reformuler le résumé avec le LLM local (sous garde)", key="llm"):
        writer = llm_client()
        if writer is None:
            st.info("LLM indisponible : rapport par template. Sur une machine de l'école, lancer `ollama serve`.")
    md = render(f, writer)
    if writer is not None and "résumé reformulé par un LLM" in md:
        st.success(f"Résumé reformulé par {writer.model}, garde-fou accepté : aucune valeur écrite par le LLM.")
    elif writer is not None:
        why = md.split("LLM non retenu : ", 1)[-1].split(")", 1)[0]
        st.warning(f"LLM non retenu ({why}) : résumé du template.")
    st.download_button("Télécharger le rapport (Markdown)", md, file_name=f"{case_id}.md")
    st.markdown(md)

st.divider()
st.subheader("Décision de l'analyste")
prev = done.get(case_id)
if prev:
    st.success(f"Décision enregistrée le {prev['date']} : {logic.LABELS[prev['decision']]}"
               + (f" ({prev['comment']})" if prev["comment"] else ""))  # fmt: skip
comment = st.text_input("Commentaire (facultatif)", key=f"comment_{case_id}")
b1, b2 = st.columns(2)
if b1.button("Valider : préparer la déclaration", type="primary", width="stretch"):
    logic.record(day, case_id, "valider", comment)
    st.rerun()
if b2.button("Classer sans suite", width="stretch"):
    logic.record(day, case_id, "classer", comment)
    st.rerun()
