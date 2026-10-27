import copy
import json
import re
import time
from pathlib import Path

import pytest

from rapport import ALERTS_DIR
from rapport.dossier import DossierError, check
from rapport.guard import numbers, unknown
from rapport.render import GuardError, render

EXAMPLE = Path(__file__).resolve().parents[1] / "rapport" / "exemples" / "C08-0002.json"
SECTIONS = ["Identification du compte", "Résumé", "Typologie", "Motifs du soupçon", "Faits chiffrés",
            "Chronologie des transactions", "Comptes liés", "Recommandation à l'analyste"]  # fmt: skip


@pytest.fixture
def f() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def sections(md: str) -> dict[str, str]:
    parts = re.split(r"^## (.+)$", md, flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts), 2)}


def figures(f: dict) -> list[str]:
    """Chiffres de l'alerte qui doivent tous apparaître dans le rapport."""
    out = [f["priority_pct"], f["max_score_pct"], f["threshold_pct"], f["typology"]["proba_pct"],
           f["period"]["start"], f["period"]["end"], f["totals"]["usd_fmt"],
           str(f["totals"]["n_ops"]), str(f["totals"]["n_accounts"]), str(f["priority_rank"])]  # fmt: skip
    for o in f["operations"]:
        out += [o["date"], o["amount_fmt"], o["usd_fmt"], o["score_pct"], str(o["index"]), o["src"], o["dst"]]
    out += [a["id"] for a in f["accounts"]]
    out += [r["text"] for r in f["reasons"]]
    return out


def test_every_figure_of_the_alert_is_in_the_report(f: dict) -> None:
    md = render(f)
    assert [x for x in figures(f) if x not in md] == []


def test_all_sections_present_and_not_empty(f: dict) -> None:
    s = sections(render(f))
    assert list(s) == SECTIONS
    assert [k for k, body in s.items() if not body.strip()] == []


def test_no_number_outside_the_facts(f: dict) -> None:
    assert unknown(render(f), f) == []


def test_guard_rejects_an_invented_number(f: dict) -> None:
    md = render(f) + "\nMontant estimé : 1 234 567,89 USD sur 42 jours."
    assert sorted(unknown(md, f)) == ["1234567,89", "42"]


def test_guard_ignores_digits_inside_identifiers() -> None:
    assert numbers("compte `0139342:80EB1E930`, le 09/09/2022 05:18, 62 956,81 USD") == [
        "0139342", "09", "09", "2022", "05", "18", "62956,81"]  # fmt: skip


def test_render_refuses_a_report_with_a_foreign_number(f: dict, monkeypatch) -> None:
    import rapport.render as r

    monkeypatch.setattr(r, "describe", lambda code: ("Définition.", "Seuil de 9999 dépassé.", []))
    with pytest.raises(GuardError, match="9999"):
        render(f)


def test_without_reasons_the_section_is_still_filled(f: dict) -> None:
    g = copy.deepcopy(f)
    g["reasons"], g["indicators"], g["evidence_lines"] = [], [], []
    g["typology"] = {"code": "OTHER", "label": "Hors typologie connue", "proba_pct": "30 %"}
    s = sections(render(g))
    assert "Aucune raison individuelle" in s["Motifs du soupçon"]
    assert all(body.strip() for body in s.values())


def test_missing_field_is_rejected(f: dict) -> None:
    del f["operations"]
    with pytest.raises(DossierError, match="operations"):
        check(f)


def test_render_time(f: dict) -> None:
    render(f)  # première fois : compilation du template
    t = time.perf_counter()
    for _ in range(20):
        render(f)
    assert (time.perf_counter() - t) / 20 < 0.05  # moins de 50 ms par rapport


@pytest.mark.skipif(not ALERTS_DIR.exists(), reason="dossiers non générés (python -m rapport.make_alerts)")
def test_every_generated_dossier_passes() -> None:
    paths = sorted(ALERTS_DIR.glob("*/C*.json"))
    for p in paths:
        g = json.loads(p.read_text(encoding="utf-8"))
        md = render(g)
        assert [x for x in figures(g) if x not in md] == [], p.name
