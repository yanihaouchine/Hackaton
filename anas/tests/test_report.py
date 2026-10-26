import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

from aml.dossier import build_dossier
from aml.guard import unknown_numbers
from aml.render import render_report
from aml.schema import AlertDossier
from aml.writers import TemplateWriter, get_writer, write_checked

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "contracts" / "examples" / "fanout_dossier.json"
RATES = {"US Dollar": 1.0, "Euro": 1.17, "Yuan": 0.15}


def rows(d: AlertDossier) -> list[dict]:
    return [
        {
            "tx_id": t.tx_id,
            "ts": t.ts,
            "from_bank": t.src.bank,
            "from_account": t.src.account,
            "to_bank": t.dst.bank,
            "to_account": t.dst.account,
            "amount_paid": t.amount_paid,
            "currency_paid": t.currency_paid,
            "amount_received": t.amount_received,
            "currency_received": t.currency_received,
            "payment_format": t.payment_format,
        }
        for t in d.transactions
    ]


@pytest.fixture
def given() -> AlertDossier:
    return AlertDossier.from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))


@pytest.fixture
def dossier(given: AlertDossier) -> AlertDossier:
    return build_dossier(given.alert, pl.DataFrame(rows(given)), given.subject_kyc, RATES)


def test_facts_match_hand_computed_example(dossier: AlertDossier) -> None:
    F = {f.key: f.text for f in dossier.facts}
    assert F["n_tx"] == "7" and F["n_counterparties"] == "7"
    assert F["span_minutes"] == "14 h 05"
    assert F["total_USD"] == "5 753 233,36 USD"
    assert F["total_EUR"] == "24 016,28 EUR"
    assert F["total_CNY"] == "35 642,49 CNY"
    assert F["n_self_fx"] == "1"


def test_dossier_ignores_future_transactions(given: AlertDossier, dossier: AlertDossier) -> None:
    future = rows(given)[-1] | {"tx_id": 99, "ts": given.alert.triggered_at + timedelta(minutes=5)}
    again = build_dossier(
        given.alert, pl.DataFrame([*rows(given), future]), given.subject_kyc, RATES
    )
    assert again == dossier


def test_template_report_passes_guard(dossier: AlertDossier) -> None:
    md = render_report(dossier, TemplateWriter())
    assert "garde-fou numérique : accepté" in md
    assert "Dispersion (fan-out)" in md


def test_guard_rejects_invented_numbers(dossier: AlertDossier) -> None:
    assert unknown_numbers("7 bénéficiaires, 5 753 233,36 USD", dossier) == []
    assert unknown_numbers("environ 5,8 millions en 42 jours", dossier) == ["5,8", "42"]


def test_llm_writer_rejected_text_falls_back(dossier: AlertDossier) -> None:
    class Liar:
        name = "fake-llm"

        def write(self, d: AlertDossier) -> dict[str, str]:
            return {"synthesis": "Montant de 1 234 567 USD.", "analysis": "Rien."}

    sections, status = write_checked(Liar(), dossier)
    assert status.startswith("rejeté")
    assert sections == TemplateWriter().write(dossier)


def test_llm_mode_falls_back_without_model() -> None:
    assert get_writer("llm").name == "template"


def test_report_needs_no_label(given: AlertDossier) -> None:
    assert "is_laundering" not in pl.DataFrame(rows(given)).columns
    assert replace(given.alert).source == "fixture_patterns"
