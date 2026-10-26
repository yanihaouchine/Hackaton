"""Rendu Markdown du rapport à partir du dossier."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from aml.config import PAYMENT_FORMATS_FR
from aml.fmt import fmt_amount, fmt_dt
from aml.schema import AlertDossier
from aml.typologies import TYPOLOGIES
from aml.writers import Writer, facts_text, reason_sentences, write_checked

TEMPLATES = Path(__file__).resolve().parents[2] / "templates"
_env = Environment(
    loader=FileSystemLoader(TEMPLATES),
    undefined=StrictUndefined,
    trim_blocks=False,
    keep_trailing_newline=True,
    autoescape=False,
)


class _Facts(dict):
    """Accès `F.cle` dans le template ; « — » si le fait n'existe pas."""

    def __getattr__(self, key: str) -> str:
        return self.get(key, "—")


def render_report(dossier: AlertDossier, writer: Writer) -> str:
    text, guard_status = write_checked(writer, dossier)
    alert_ids = set(dossier.alert.tx_ids)
    rows = [
        {
            "tx_id": t.tx_id,
            "flag": t.tx_id in alert_ids,
            "ts": fmt_dt(t.ts),
            "src": t.src.key(),
            "dst": t.dst.key(),
            "paid": fmt_amount(t.amount_paid, t.currency_paid),
            "received": fmt_amount(t.amount_received, t.currency_received),
            "format": PAYMENT_FORMATS_FR.get(t.payment_format, t.payment_format),
        }
        for t in dossier.transactions
    ]
    return _env.get_template("rapport_fr.md.j2").render(
        alert=dossier.alert,
        F=_Facts(facts_text(dossier)),
        facts=dossier.facts,
        totals=[
            f for f in dossier.facts if f.key.startswith("total_") and f.key != "total_usd_estimate"
        ],
        typ=TYPOLOGIES[dossier.alert.typology],
        kyc=dossier.subject_kyc,
        text=text,
        rows=rows,
        reasons=reason_sentences(dossier),
        writer_name=writer.name,
        guard_status=guard_status,
    )
