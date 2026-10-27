import json
from pathlib import Path

import pytest

from interface import logic

EXAMPLE = Path(__file__).resolve().parents[1] / "rapport" / "exemples" / "C08-0002.json"


def test_decisions_last_one_wins(tmp_path: Path) -> None:
    log = tmp_path / "d.jsonl"
    logic.record("2022-09-09", "C08-0002", "valider", path=log)
    logic.record("2022-09-09", "C08-0002", "classer", "erreur corrigée", path=log)
    logic.record("2022-09-10", "C10-0001", "valider", path=log)
    d = logic.decisions("2022-09-09", path=log)
    assert list(d) == ["C08-0002"] and d["C08-0002"]["decision"] == "classer"
    assert len(log.read_text().splitlines()) == 3  # journal : rien n'est effacé


def test_unknown_decision_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        logic.record("2022-09-09", "C08-0002", "supprimer", path=tmp_path / "d.jsonl")


def test_graph_has_one_edge_per_operation_and_highlights_main_account() -> None:
    f = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    dot = logic.case_dot(f, "0139342:80EB1E930")
    assert dot.count("->") == len(f["operations"])
    assert all(f'"{a["id"]}" [label=' in dot for a in f["accounts"])
    assert dot.count("#eb6834") == 1  # seul le compte principal est en évidence


def test_interface_does_not_import_the_detection() -> None:
    src = (Path(logic.__file__).parent / "app.py").read_text(encoding="utf-8")
    assert "import aml" not in src and "from aml" not in src and "nn." not in src
