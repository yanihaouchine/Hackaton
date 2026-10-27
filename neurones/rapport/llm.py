"""Reformulation du résumé par un petit LLM local, sous garde stricte (principe 2 de CLAUDE.md).

1. Le LLM ne voit aucune valeur : chaque nombre du texte à reformuler est remplacé par un marqueur ⟦Fn⟧.
2. Sa réponse est rejetée si elle contient un chiffre, un marqueur inconnu, ou si elle perd le montant total.
3. Sinon, le code remet les vraies valeurs à la place des marqueurs ; le garde-fou du rapport revérifie tout.
En cas d'échec (serveur absent, délai dépassé, réponse rejetée), le résumé du template est gardé.

Serveur : toute API compatible OpenAI (`/v1/chat/completions`), par exemple Ollama sur les machines de l'école
(`ollama serve`, port 11434) ou `llama-server`. Variables : AML_LLM_URL, AML_LLM_MODEL.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request

from rapport.guard import NUMBER, unknown

DEFAULT_URL = "http://localhost:11434"  # Ollama
MARK = re.compile(r"⟦F(\d+)⟧")
SYSTEM = (
    "Tu es analyste conformité dans une banque française. Tu reformules le résumé d'un projet de déclaration de "
    "soupçon, en français sobre et factuel, en un seul paragraphe de trois à cinq phrases. Règles absolues : "
    "n'écris aucun chiffre ; recopie chaque marqueur de la forme ⟦F suivi d'un numéro⟧ exactement tel quel, à "
    "l'endroit où sa valeur doit apparaître ; n'ajoute aucun fait qui ne figure pas dans le texte. "
    "Réponds uniquement par le paragraphe."
)
MAX_REASONS = 3


class LlmUnavailable(RuntimeError):
    pass


class OpenAICompatible:
    """Client minimal (bibliothèque standard seulement) d'un serveur LLM local compatible OpenAI."""

    def __init__(self, url: str | None = None, model: str | None = None, timeout: float = 120.0):
        self.url = (url or os.environ.get("AML_LLM_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.model = model or os.environ.get("AML_LLM_MODEL") or self._first_model()

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.url + path, timeout=5) as r:
                return json.load(r)
        except OSError as e:
            raise LlmUnavailable(f"serveur LLM injoignable ({self.url})") from e

    def _first_model(self) -> str:
        models = [m["id"] for m in self._get("/v1/models").get("data", [])]
        if not models:
            raise LlmUnavailable("aucun modèle servi")
        return models[0]

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "temperature": 0, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}  # fmt: skip
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.load(r)["choices"][0]["message"]["content"]
        except OSError as e:
            raise LlmUnavailable(f"pas de réponse du LLM ({e.__class__.__name__})") from e
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # modèles « à raisonnement »


def mask(text: str, wholes: tuple[str, ...] = ()) -> tuple[str, dict[str, str]]:
    """Remplace chaque valeur par un marqueur ⟦Fn⟧ : d'abord les valeurs entières du dossier (dates, montants),
    puis chaque nombre restant. Renvoie le texte masqué et la table marqueur → valeur."""
    values: dict[str, str] = {}

    def key_for(v: str) -> str:
        key = f"⟦F{len(values) + 1}⟧"
        values[key] = v
        return key

    for w in sorted(set(wholes), key=len, reverse=True):
        if w and w in text:
            text = text.replace(w, key_for(w))
    return NUMBER.sub(lambda m: key_for(m.group(0)), text), values


def source_text(f: dict) -> str:
    """Texte confié au LLM : résumé du template et principales raisons (sans identifiant de compte)."""
    from rapport.render import summary

    reasons = [r["text"] for r in f["reasons"] if not r.get("context") and "`" not in r["text"]][:MAX_REASONS]
    return summary(f) + (" Principaux motifs : " + " ; ".join(reasons) + "." if reasons else "")


def check_answer(answer: str, values: dict[str, str], required: str) -> str | None:
    """Raison du rejet, ou None si la réponse est acceptée."""
    if not answer or len(answer) > 2000:
        return "réponse vide ou trop longue"
    used = {m.group(0) for m in MARK.finditer(answer)}
    if used - set(values):
        return "marqueur inconnu"
    if re.search(r"\d", MARK.sub("", answer)):
        return "chiffre écrit par le LLM"
    if required not in used:
        return "montant total absent"
    return None


def rewrite(f: dict, writer) -> tuple[str, dict]:
    """(résumé, statut). Le statut dit d'où vient le texte (llm ou template), pourquoi, et en combien de temps."""
    from rapport.render import summary

    fallback = summary(f)
    total = f["totals"]["usd_fmt"]
    masked, values = mask(source_text(f), (f["period"]["start"], f["period"]["end"], total, f["priority_pct"]))
    required = next(k for k, v in values.items() if v == total)  # le montant total doit rester dans le résumé
    status = {"source": "template", "model": getattr(writer, "model", "?"), "seconds": 0.0, "reason": ""}
    t = time.perf_counter()
    try:
        answer = writer.complete(SYSTEM, masked)
    except LlmUnavailable as e:
        status.update(reason=str(e), seconds=round(time.perf_counter() - t, 2))
        return fallback, status
    status["seconds"] = round(time.perf_counter() - t, 2)
    reason = check_answer(answer, values, required)
    if reason is None:
        text = MARK.sub(lambda m: values[m.group(0)], answer)
        if unknown(text, f):  # double vérification après réinjection des valeurs
            reason = "nombre absent du dossier après réinjection"
        else:
            status.update(source="llm")
            return text, status
    status["reason"] = reason
    return fallback, status
