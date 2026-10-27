"""LLM local, sous garde stricte, pour deux rôles : reformuler l'analyse du rapport et répondre à l'analyste.

Garde : le LLM ne voit jamais une valeur. Chaque nombre, date, montant ou identifiant du dossier est
remplacé par un jeton ⟦Vn⟧ ; la réponse est rejetée si elle contient un chiffre ou un jeton inconnu,
sinon le code réinjecte les vraies valeurs. Backend : llama_cpp (in-process) si disponible, sinon un
serveur llama-server compatible OpenAI (AML_LLM_URL), sinon aucun : les templates prennent le relais.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path

from aml.config import REPO

SLOT_RE = re.compile(
    r"`[^`]+`"  # identifiants de comptes
    r"|\d{2}/\d{2}/\d{4}(?: \d{2}:\d{2})?"  # dates
    r"|\d[\d\u202f\u00a0 ]*(?:,\d+)?\s?(?:USD\b|%|min\b|h\b|j\b)"  # quantités avec unité
    r"|\d[\d\u202f\u00a0]*(?:,\d+)?(?![\w:#./-])"  # nombres à la française (1 234,5)
    r"|[A-Za-z]*\d[\w:#./-]*"  # tout autre jeton contenant un chiffre (n° de banque, codes…)
)
TOKEN_RE = re.compile(r"⟦[A-Z]+_\d+⟧")
# ordre de préférence : le premier présent est chargé par défaut. Mesure (results/llm_bench_ORC_x86_64.json,
# jetons typés) : Qwen3.5-0.8B accepté à 100 % en 25 s, Qwen3.5-2B à 90 % en 47 s, LFM2.5-1.2B à 0 %.
DEFAULT_MODELS = [
    "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf",
    "models/qwen3.5-2b/Qwen3.5-2B-Q4_K_M.gguf",
    "models/lfm2.5-1.2b/LFM2.5-1.2B-Instruct-Q4_K_M.gguf",
]


def _kind(v: str) -> str:
    """Type d'un jeton, visible du LLM : il sait ce que représente chaque valeur sans la voir."""
    if v.startswith("`"):
        return "COMPTE"
    if "/" in v and v[:2].isdigit():
        return "DATE"
    if v.endswith("USD"):
        return "MONTANT"
    if v.endswith("%"):
        return "POURCENT"
    if v.endswith(("min", "h", "j")):
        return "DUREE"
    if v.replace("\u202f", "").replace("\u00a0", "").replace(" ", "").replace(",", "").isdigit():
        return "NOMBRE"
    return "CODE"


def slotify(texts: list[str], slots: dict[str, str] | None = None) -> tuple[list[str], dict[str, str]]:
    """Remplace toute valeur par un jeton ; renvoie les textes masqués et la table jeton → valeur."""
    slots = dict(slots or {})
    rev = {v: k for k, v in slots.items()}

    def sub(m):
        v = m.group(0)
        if v not in rev:
            k = f"⟦{_kind(v)}_{len(slots) + 1}⟧"
            slots[k] = v
            rev[v] = k
        return rev[v]

    return [SLOT_RE.sub(sub, t) for t in texts], slots


VARIANT_RE = re.compile(r"[⟦⟨〈<\[{«]\s*([A-Z]+)\s*_\s*(\d+)\s*[⟧⟩〉>\]}»]")


def guard(text: str, slots: dict[str, str], min_coverage: float = 0.0) -> tuple[bool, str]:
    """Rejette tout chiffre écrit par le LLM, tout jeton inconnu et (si demandé) un texte qui reprend moins
    de `min_coverage` des jetons fournis ; sinon réinjecte les valeurs exactes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = VARIANT_RE.sub(lambda m: f"⟦{m.group(1)}_{m.group(2)}⟧", text)  # crochets approximatifs
    tokens = TOKEN_RE.findall(text)
    if any(t not in slots for t in tokens):
        return False, "jeton inconnu"
    if re.search(r"\d", TOKEN_RE.sub("", text)):
        return False, "chiffre écrit par le LLM"
    if len(text) < 40:
        return False, "réponse vide ou trop courte"
    if slots and len(set(tokens)) < min_coverage * len(slots):
        return False, f"texte peu ancré ({len(set(tokens))}/{len(slots)} valeurs reprises)"
    for t in set(tokens):
        text = text.replace(t, slots[t])
    return True, text


class LLM:
    def __init__(self, model_path: str | None = None, n_threads: int = 4, n_ctx: int = 4096):
        self.kind, self.name, self._llm = "aucun", None, None
        self.url = os.environ.get("AML_LLM_URL")
        path = model_path or os.environ.get("AML_LLM_MODEL")
        if path is None:
            path = next((p for p in DEFAULT_MODELS if (REPO / p).exists()), None)
        if path:
            try:
                from llama_cpp import Llama

                full = Path(path) if Path(path).is_absolute() else REPO / path
                self._llm = Llama(model_path=str(full), n_ctx=n_ctx, n_threads=n_threads, seed=42, verbose=False)
                self.kind, self.name = "llama_cpp", full.name
            except (ImportError, ValueError, OSError):
                self._llm = None
        if self._llm is None and self.url:
            self.kind, self.name = "http", self.url

    @property
    def available(self) -> bool:
        return self.kind != "aucun"

    def chat(self, system: str, user: str, max_tokens: int = 400) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.kind == "llama_cpp":
            r = self._llm.create_chat_completion(messages=msgs, max_tokens=max_tokens, temperature=0.2, seed=42)
            return r["choices"][0]["message"]["content"] or ""
        if self.kind == "http":
            body = json.dumps({"messages": msgs, "max_tokens": max_tokens, "temperature": 0.2}).encode()
            req = urllib.request.Request(self.url.rstrip("/") + "/v1/chat/completions", data=body,
                                         headers={"Content-Type": "application/json"})  # fmt: skip
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
        raise RuntimeError("aucun LLM disponible")


SYSTEM_REPORT = (
    "Tu es analyste conformité LCB-FT dans une banque française. Tu rédiges en français, ton factuel et "
    "prudent. Règles absolues : tu n'écris AUCUN chiffre ; pour toute valeur (montant, date, nombre, compte, "
    "pourcentage), tu recopies exactement le jeton fourni (⟦MONTANT_3⟧, ⟦DATE_1⟧, ⟦COMPTE_2⟧…) en respectant "
    "son type ; tu n'ajoutes aucun fait absent des éléments fournis ; tu ne conclus pas à la culpabilité (on "
    "parle de soupçon)."
)


def write_analysis(llm: LLM, facts: dict, min_coverage: float = 0.5) -> dict:
    """Reformulation fluide de l'analyse rédigée par template (tâche simple et vérifiable pour un petit
    modèle) ; rejet si un chiffre apparaît, si un jeton est inventé ou si moins de la moitié des valeurs
    sont reprises. Repli : le texte du template, inchangé."""
    src = [facts["analysis_text"], *facts["evidence_lines"][:2]]
    masked, slots = slotify(src)
    user = (
        "Texte à reformuler (les valeurs sont masquées par des jetons) :\n" + " ".join(masked) + "\n\n"
        "Reformule ce texte en un paragraphe fluide et professionnel de 4 à 6 phrases pour une déclaration de "
        "soupçon. Recopie chaque jeton (⟦TYPE_n⟧) exactement, sans le modifier, à une place qui correspond à "
        "son type (une date reste une date, un compte reste un compte). N'écris aucun chiffre. N'ajoute aucun fait."
    )
    t0 = time.perf_counter()
    try:
        raw = llm.chat(SYSTEM_REPORT, user)
    except Exception as exc:  # LLM indisponible ou en erreur : repli template
        return {"text": facts["analysis_text"], "source": "template", "reason": f"erreur LLM : {exc}",
                "seconds": time.perf_counter() - t0}  # fmt: skip
    ok, text = guard(raw, slots, min_coverage=min_coverage)
    return {"text": text if ok else facts["analysis_text"], "source": f"llm:{llm.name}" if ok else "template",
            "reason": "accepté" if ok else text, "seconds": time.perf_counter() - t0, "raw": raw,
            "n_slots": len(slots)}  # fmt: skip


def bench(dossiers: list[dict], models: list[str] | None = None, n_threads: int = 4) -> dict:
    """Compare les LLM locaux sur des dossiers réels : chargement, latence par rapport, taux d'acceptation
    par la garde, valeurs reprises. Le meilleur compromis devient le modèle par défaut (mesure, pas avis)."""
    import platform

    out = {"n_dossiers": len(dossiers), "n_threads": n_threads, "machine": platform.machine(), "models": {}}
    for path in models or DEFAULT_MODELS:
        if not (REPO / path).exists():
            continue
        t0 = time.perf_counter()
        llm = LLM(path, n_threads=n_threads)
        load_s = time.perf_counter() - t0
        if not llm.available:
            continue
        runs = [write_analysis(llm, f) for f in dossiers]
        acc = [r for r in runs if r["source"] != "template"]
        out["models"][Path(path).name] = {
            "load_s": load_s,
            "latency_s_median": float(sorted(r["seconds"] for r in runs)[len(runs) // 2]),
            "acceptance_rate": len(acc) / len(runs),
            "rejections": [r["reason"] for r in runs if r["source"] == "template"],
            "example": acc[0]["text"] if acc else None,
        }
        del llm
    return out
