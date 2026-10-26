import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from aml.schema import AlertDossier, ContractError, Fact

EXAMPLE = Path(__file__).resolve().parents[1] / "contracts" / "examples" / "fanout_dossier.json"


@pytest.fixture
def dossier() -> AlertDossier:
    return AlertDossier.from_dict(json.loads(EXAMPLE.read_text(encoding="utf-8")))


def test_example_loads(dossier: AlertDossier) -> None:
    assert dossier.alert.typology == "fan_out"
    assert len(dossier.transactions) == 8
    assert len(dossier.facts) == 8


def test_round_trip(dossier: AlertDossier) -> None:
    assert AlertDossier.from_dict(dossier.to_dict()) == dossier


def test_bank_code_keeps_leading_zeros(dossier: AlertDossier) -> None:
    assert dossier.alert.subject.bank == "021174"
    assert dossier.transactions[4].dst.key() == "00220/8007A5B70"


def test_rejects_future_transaction(dossier: AlertDossier) -> None:
    late = replace(dossier.transactions[-1], ts=dossier.alert.triggered_at + timedelta(minutes=1))
    with pytest.raises(ContractError, match="postérieures"):
        replace(dossier, transactions=(*dossier.transactions[:-1], late))


def test_rejects_unknown_typology(dossier: AlertDossier) -> None:
    with pytest.raises(ContractError, match="typologie"):
        replace(dossier.alert, typology="smurfing")


def test_rejects_trigger_outside_tx_ids(dossier: AlertDossier) -> None:
    with pytest.raises(ContractError, match="déclenchante"):
        replace(dossier.alert, trigger_tx_id=999)


def test_rejects_duplicate_fact_id(dossier: AlertDossier) -> None:
    with pytest.raises(ContractError, match="double"):
        replace(dossier, facts=(*dossier.facts, Fact("F1", "dup", 0, "0")))
