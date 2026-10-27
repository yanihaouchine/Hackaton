import json
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from rapport.guard import unknown
from rapport.llm import LlmUnavailable, OpenAICompatible, facts_for_llm, rewrite
from rapport.render import render, summary

EXAMPLE = Path(__file__).resolve().parents[1] / "rapport" / "exemples" / "C08-0002.json"


@pytest.fixture
def f() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class Fake:
    """Faux LLM : renvoie une réponse construite à partir du texte masqué qu'il reçoit."""

    model = "faux-llm"

    def __init__(self, answer):
        self.answer = answer
        self.seen = ""

    def complete(self, system: str, user: str) -> str:
        self.seen = user
        return self.answer(user)


def good(user: str) -> str:
    return "Du ⟦DEBUT⟧ au ⟦FIN⟧, plusieurs comptes ont réalisé des opérations signalées pour un total de ⟦TOTAL⟧."


def test_the_llm_never_sees_a_number(f: dict) -> None:
    w = Fake(good)
    rewrite(f, w)
    assert w.seen and not re.search(r"\d", re.sub(r"⟦[A-Z]+⟧", "", w.seen))


def test_good_answer_is_accepted_and_values_reinjected(f: dict) -> None:
    text, status = rewrite(f, Fake(good))
    assert status["source"] == "llm"
    assert f["totals"]["usd_fmt"] in text and f["period"]["start"] in text
    assert unknown(text, f) == []


def test_answer_with_a_digit_is_rejected(f: dict) -> None:
    text, status = rewrite(f, Fake(lambda u: good(u) + " Soit environ 63 000 dollars."))
    assert status == {**status, "source": "template", "reason": "chiffre écrit par le LLM"}
    assert text == summary(f)


def test_number_in_words_is_rejected(f: dict) -> None:
    _, status = rewrite(f, Fake(lambda u: good(u) + " Trois comptes sont concernés."))
    assert status["reason"] == "nombre écrit en lettres par le LLM"


def test_number_word_from_the_facts_is_allowed(f: dict) -> None:
    _, status = rewrite(f, Fake(lambda u: good(u) + " C'est la première relation entre ces deux comptes."))
    assert status["source"] == "llm"  # « deux » figure dans les raisons du dossier


def test_unknown_marker_is_rejected(f: dict) -> None:
    _, status = rewrite(f, Fake(lambda u: good(u) + " Voir ⟦ZZ⟧."))
    assert status["reason"] == "marqueur inconnu"


def test_answer_without_the_total_is_rejected(f: dict) -> None:
    _, status = rewrite(f, Fake(lambda u: "Des opérations suspectes ont été relevées."))
    assert status["reason"] == "marqueur obligatoire absent"


def test_unreachable_server_falls_back_to_template(f: dict) -> None:
    class Down(Fake):
        def complete(self, system: str, user: str) -> str:
            raise LlmUnavailable("serveur LLM injoignable")

    md = render(f, Down(good))
    assert "LLM non retenu : serveur LLM injoignable" in md
    assert summary(f) in md


def test_report_with_llm_passes_the_guard(f: dict) -> None:
    md = render(f, Fake(good))
    assert "résumé reformulé par un LLM local (faux-llm)" in md
    assert unknown(md, {**f, "redaction": {"model": "faux-llm"}}) == []


def test_markers_carry_meaning_and_values_round_trip(f: dict) -> None:
    text, values = facts_for_llm(f)
    assert values["⟦TOTAL⟧"] == f["totals"]["usd_fmt"] and values["⟦DEBUT⟧"] == f["period"]["start"]
    assert not re.search(r"\d", re.sub(r"⟦[A-Z]+⟧", "", text))  # aucune valeur en clair
    assert "7,3" not in text  # les raisons chiffrées ne sont pas confiées au LLM


def test_client_against_a_local_openai_compatible_server(f: dict) -> None:
    """Le client parle bien le protocole d'Ollama / llama-server (faux serveur HTTP local)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self._send({"data": [{"id": "qwen-test"}]})

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            answer = "<think>brouillon</think>" + good(body["messages"][1]["content"])
            self._send({"choices": [{"message": {"content": answer}}]})

        def _send(self, obj):
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        w = OpenAICompatible(f"http://127.0.0.1:{srv.server_port}")
        assert w.model == "qwen-test"
        text, status = rewrite(f, w)
        assert status["source"] == "llm" and "brouillon" not in text
    finally:
        srv.shutdown()


def test_no_server_raises_unavailable() -> None:
    with pytest.raises(LlmUnavailable):
        OpenAICompatible("http://127.0.0.1:9")
