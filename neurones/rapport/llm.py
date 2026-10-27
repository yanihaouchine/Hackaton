"""Reformulation du résumé par un petit LLM local, sous garde stricte (principe 2 de CLAUDE.md).

1. Le LLM ne voit aucune valeur : chaque valeur est remplacée par un marqueur nommé (⟦DEBUT⟧, ⟦TOTAL⟧…).
2. Sa réponse est rejetée si elle contient un chiffre, un nombre en lettres, un marqueur inconnu, ou si elle perd
   un marqueur obligatoire (dates de début et de fin, montant total).
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
PREFERRED = ("qwen3:1.7b", "ministral-3:3b")  # meilleurs taux d'acceptation mesurés (results/rapport_bench_llm_*)
MARK = re.compile(r"⟦([A-Z]+)⟧")  # noms en lettres : un modèle ne peut pas les prendre pour des valeurs
SYSTEM = (
    "Tu es analyste conformité dans une banque française. Tu reformules le résumé d'un projet de déclaration de "
    "soupçon, en français sobre et factuel, en un seul paragraphe de trois à cinq phrases. Règles absolues : "
    "n'écris aucun chiffre ni aucun nombre en lettres ; recopie chaque marqueur entre crochets, comme ⟦TOTAL⟧ "
    "ou ⟦DEBUT⟧, exactement tel quel, à l'endroit où sa valeur doit apparaître ; "
    "garde tous les marqueurs obligatoires ; "
    "n'ajoute aucun fait qui ne figure pas dans le texte. "
    "Réponds uniquement par le paragraphe."
)
MAX_REASONS = 3
# Nombres écrits en lettres (« un » et « une » exclus : ce sont aussi des articles)
NUMBER_WORDS = re.compile(
    r"\b(deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingt|trente|"
    r"quarante|cinquante|soixante|cents?|milles?|millions?|milliards?|moitié|double|triple|quadruple|"
    r"dizaines?|douzaines?|centaines?|milliers?)\b",
    re.I,
)


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
        return next((m for m in PREFERRED if m in models), models[0])

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "temperature": 0, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}  # fmt: skip
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})  # fmt: skip
        for attempt in (1, 2):  # serveur partagé : une seconde tentative si le premier appel échoue
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    text = json.load(r)["choices"][0]["message"]["content"]
                break
            except OSError as e:
                if attempt == 2:
                    raise LlmUnavailable(f"pas de réponse du LLM ({e.__class__.__name__})") from e
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # modèles « à raisonnement »


REQUIRED = ("⟦DEBUT⟧", "⟦FIN⟧", "⟦TOTAL⟧")


def facts_for_llm(f: dict) -> tuple[str, dict[str, str]]:
    """Texte confié au LLM et table marqueur → valeur. Les marqueurs disent ce qu'ils désignent (⟦TOTAL⟧ et non
    ⟦C⟧) pour que le modèle ne les place pas au mauvais endroit ; seules les raisons sans chiffre sont données."""
    values = {
        "⟦DEBUT⟧": f["period"]["start"],
        "⟦FIN⟧": f["period"]["end"],
        "⟦COMPTES⟧": str(f["totals"]["n_accounts"]),
        "⟦OPERATIONS⟧": str(f["totals"]["n_ops"]),
        "⟦TOTAL⟧": f["totals"]["usd_fmt"],
        "⟦RANG⟧": str(f["priority_rank"]),
        "⟦PRIORITE⟧": f["priority_pct"],
    }
    text = (
        "Entre le ⟦DEBUT⟧ et le ⟦FIN⟧, ⟦COMPTES⟧ comptes ont réalisé ⟦OPERATIONS⟧ opérations signalées, pour un "
        "total de ⟦TOTAL⟧ (contre-valeur en dollars). Le cas est classé au rang ⟦RANG⟧ de la journée (priorité "
        f"⟦PRIORITE⟧) et se rapproche de la typologie « {f['typology']['label']} »."
    )
    reasons = [r["text"] for r in f["reasons"] if not NUMBER.search(r["text"]) and "`" not in r["text"]]
    if reasons[:MAX_REASONS]:
        text += " Principaux motifs : " + " ; ".join(reasons[:MAX_REASONS]) + "."
    return text + "\n\nMarqueurs obligatoires : " + ", ".join(REQUIRED) + ".", values


def check_answer(answer: str, values: dict[str, str], required: list[str], source: str = "") -> str | None:
    """Raison du rejet, ou None si la réponse est acceptée. `source` : texte confié au LLM (ses mots sont permis)."""
    if not answer or len(answer) > 2000:
        return "réponse vide ou trop longue"
    used = {m.group(0) for m in MARK.finditer(answer)}
    if used - set(values):
        return "marqueur inconnu"
    if re.search(r"\d", MARK.sub("", answer)):
        return "chiffre écrit par le LLM"
    allowed_words = {w.lower() for w in NUMBER_WORDS.findall(source)}  # ex. « entre ces deux comptes »
    if {w.lower() for w in NUMBER_WORDS.findall(answer)} - allowed_words:
        return "nombre écrit en lettres par le LLM"
    if not set(required) <= used:
        return "marqueur obligatoire absent"
    return None


def rewrite(f: dict, writer) -> tuple[str, dict]:
    """(résumé, statut). Le statut dit d'où vient le texte (llm ou template), pourquoi, et en combien de temps."""
    from rapport.render import summary

    fallback = summary(f)
    masked, values = facts_for_llm(f)
    status = {"source": "template", "model": getattr(writer, "model", "?"), "seconds": 0.0, "reason": ""}
    t = time.perf_counter()
    try:
        answer = writer.complete(SYSTEM, masked)
    except LlmUnavailable as e:
        status.update(reason=str(e), seconds=round(time.perf_counter() - t, 2))
        return fallback, status
    status["seconds"] = round(time.perf_counter() - t, 2)
    reason = check_answer(answer, values, list(REQUIRED), masked)
    if reason is None:
        text = MARK.sub(lambda m: values[m.group(0)], answer)
        if unknown(text, f):  # double vérification après réinjection des valeurs
            reason = "nombre absent du dossier après réinjection"
        else:
            status.update(source="llm")
            return text, status
    status["reason"] = reason
    return fallback, status
