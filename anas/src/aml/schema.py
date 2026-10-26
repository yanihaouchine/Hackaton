"""Contrat d'échange entre la détection et le module de rapport (v0.1).

- `Alert` : produite par la détection (codes et valeurs, pas de phrases).
- `AlertDossier` : construit par le module de rapport à partir de l'alerte et
  de l'historique antérieur du compte. Tout nombre du rapport vient d'un `Fact`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

SCHEMA_VERSION = "0.1"

TYPOLOGIES = frozenset(
    {
        "fan_in",
        "fan_out",
        "cycle",
        "gather_scatter",
        "scatter_gather",
        "bipartite",
        "stack",
        "random",
        "structuring",
        "dormant",
        "rapid_chain",
        "unknown",
    }
)
SCORE_KINDS = frozenset({"rule", "proba"})
SOURCES = frozenset({"detector", "fixture_patterns"})
FACT_ID = re.compile(r"F\d+")

FactValue = int | float | str | bool


class ContractError(ValueError):
    """Donnée qui ne respecte pas le contrat."""


def _parse_ts(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as e:
        raise ContractError(f"horodatage invalide : {value!r}") from e


@dataclass(frozen=True)
class AccountRef:
    """Un compte s'identifie par le couple (banque, compte)."""

    bank: str
    account: str

    def __post_init__(self) -> None:
        # Code banque gardé en chaîne : les zéros en tête comptent ("021174").
        if not isinstance(self.bank, str) or not self.bank.isdigit():
            raise ContractError(f"code banque invalide : {self.bank!r}")
        if not isinstance(self.account, str) or not self.account:
            raise ContractError(f"numéro de compte invalide : {self.account!r}")

    def key(self) -> str:
        return f"{self.bank}/{self.account}"

    def to_dict(self) -> dict[str, str]:
        return {"bank": self.bank, "account": self.account}


@dataclass(frozen=True)
class Reason:
    """Indicateur déclenché : code, valeur observée, seuil, fenêtre."""

    code: str
    value: float
    threshold: float | None = None
    window_h: float | None = None


@dataclass(frozen=True)
class Alert:
    alert_id: str
    triggered_at: datetime
    subject: AccountRef
    trigger_tx_id: int
    tx_ids: tuple[int, ...]
    score: float
    score_kind: str
    priority_rank: int
    typology: str
    reasons: tuple[Reason, ...]
    detector_name: str
    detector_version: str
    source: str = "detector"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"version de schéma non gérée : {self.schema_version}")
        if self.typology not in TYPOLOGIES:
            raise ContractError(f"typologie inconnue : {self.typology!r}")
        if self.score_kind not in SCORE_KINDS:
            raise ContractError(f"type de score inconnu : {self.score_kind!r}")
        if self.source not in SOURCES:
            raise ContractError(f"source inconnue : {self.source!r}")
        if not self.reasons:
            raise ContractError("une alerte doit avoir au moins une raison")
        if self.trigger_tx_id not in self.tx_ids:
            raise ContractError("la transaction déclenchante doit figurer dans tx_ids")
        if len(set(self.tx_ids)) != len(self.tx_ids):
            raise ContractError("tx_ids contient des doublons")
        if self.priority_rank < 1:
            raise ContractError("priority_rank commence à 1")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alert:
        try:
            return cls(
                alert_id=d["alert_id"],
                triggered_at=_parse_ts(d["triggered_at"]),
                subject=AccountRef(**d["subject"]),
                trigger_tx_id=int(d["trigger_tx_id"]),
                tx_ids=tuple(int(t) for t in d["tx_ids"]),
                score=float(d["score"]),
                score_kind=d["score_kind"],
                priority_rank=int(d["priority_rank"]),
                typology=d["typology"],
                reasons=tuple(Reason(**r) for r in d["reasons"]),
                detector_name=d["detector"]["name"],
                detector_version=d["detector"]["version"],
                source=d.get("source", "detector"),
                schema_version=d.get("schema_version", SCHEMA_VERSION),
            )
        except (KeyError, TypeError) as e:
            raise ContractError(f"alerte invalide : {e}") from e

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alert_id": self.alert_id,
            "triggered_at": self.triggered_at.isoformat(),
            "subject": self.subject.to_dict(),
            "trigger_tx_id": self.trigger_tx_id,
            "tx_ids": list(self.tx_ids),
            "score": self.score,
            "score_kind": self.score_kind,
            "priority_rank": self.priority_rank,
            "typology": self.typology,
            "reasons": [vars(r) for r in self.reasons],
            "detector": {"name": self.detector_name, "version": self.detector_version},
            "source": self.source,
        }


@dataclass(frozen=True)
class Transaction:
    tx_id: int
    ts: datetime
    src: AccountRef
    dst: AccountRef
    amount_paid: float
    currency_paid: str
    amount_received: float
    currency_received: str
    payment_format: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Transaction:
        return cls(
            tx_id=int(d["tx_id"]),
            ts=_parse_ts(d["ts"]),
            src=AccountRef(**d["src"]),
            dst=AccountRef(**d["dst"]),
            amount_paid=float(d["amount_paid"]),
            currency_paid=d["currency_paid"],
            amount_received=float(d["amount_received"]),
            currency_received=d["currency_received"],
            payment_format=d["payment_format"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tx_id": self.tx_id,
            "ts": self.ts.isoformat(),
            "src": self.src.to_dict(),
            "dst": self.dst.to_dict(),
            "amount_paid": self.amount_paid,
            "currency_paid": self.currency_paid,
            "amount_received": self.amount_received,
            "currency_received": self.currency_received,
            "payment_format": self.payment_format,
        }


@dataclass(frozen=True)
class Fact:
    """Fait calculé par le code. `text` est la forme française injectée dans le rapport."""

    id: str
    key: str
    value: FactValue
    text: str


@dataclass(frozen=True)
class SubjectKyc:
    """Ce que `accounts.csv` dit du compte (substitut de KYC)."""

    bank_name: str
    entity_id: str
    entity_name: str
    other_accounts: tuple[AccountRef, ...] = ()


@dataclass(frozen=True)
class AlertDossier:
    alert: Alert
    subject_kyc: SubjectKyc | None
    transactions: tuple[Transaction, ...]
    facts: tuple[Fact, ...]
    history_until: datetime

    def __post_init__(self) -> None:
        # Flux causal : rien de postérieur au déclenchement.
        if self.history_until > self.alert.triggered_at:
            raise ContractError("history_until dépasse le déclenchement de l'alerte")
        late = [t.tx_id for t in self.transactions if t.ts > self.alert.triggered_at]
        if late:
            raise ContractError(f"transactions postérieures au déclenchement : {late}")
        known = {t.tx_id for t in self.transactions}
        if len(known) != len(self.transactions):
            raise ContractError("transactions en double dans le dossier")
        missing = [i for i in self.alert.tx_ids if i not in known]
        if missing:
            raise ContractError(f"transactions de l'alerte absentes du dossier : {missing}")
        ids = [f.id for f in self.facts]
        if len(set(ids)) != len(ids):
            raise ContractError("identifiants de faits en double")
        bad = [i for i in ids if not FACT_ID.fullmatch(i)]
        if bad:
            raise ContractError(f"identifiants de faits mal formés : {bad}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AlertDossier:
        try:
            kyc = d.get("subject_kyc")
            return cls(
                alert=Alert.from_dict(d["alert"]),
                subject_kyc=None
                if kyc is None
                else SubjectKyc(
                    bank_name=kyc["bank_name"],
                    entity_id=kyc["entity_id"],
                    entity_name=kyc["entity_name"],
                    other_accounts=tuple(AccountRef(**a) for a in kyc.get("other_accounts", [])),
                ),
                transactions=tuple(Transaction.from_dict(t) for t in d["transactions"]),
                facts=tuple(Fact(**f) for f in d["facts"]),
                history_until=_parse_ts(d["history_until"]),
            )
        except (KeyError, TypeError) as e:
            raise ContractError(f"dossier invalide : {e}") from e

    def to_dict(self) -> dict[str, Any]:
        kyc = self.subject_kyc
        return {
            "alert": self.alert.to_dict(),
            "subject_kyc": None
            if kyc is None
            else {
                "bank_name": kyc.bank_name,
                "entity_id": kyc.entity_id,
                "entity_name": kyc.entity_name,
                "other_accounts": [a.to_dict() for a in kyc.other_accounts],
            },
            "transactions": [t.to_dict() for t in self.transactions],
            "facts": [vars(f) for f in self.facts],
            "history_until": self.history_until.isoformat(),
        }
