"""Garde du LLM : aucune valeur visible par le modèle, aucun chiffre accepté en sortie."""

import re

from aml.llm import guard, slotify


def test_slotify_masks_every_value():
    texts = ["Du 09/09/2022 14:05 au 10/09/2022 08:00, `0071:804ABCE90` a payé 14 bénéficiaires "
             "pour 12 345,67 USD (99,7 %) via Germany Bank #4507."]  # fmt: skip
    masked, slots = slotify(texts)
    assert not re.search(r"\d", re.sub(r"⟦[A-Z]+_\d+⟧", "", masked[0]))
    kinds = {k.split("_")[0].strip("⟦") for k in slots}
    assert {"DATE", "COMPTE", "MONTANT", "POURCENT", "NOMBRE"} <= kinds
    assert "`0071:804ABCE90`" in slots.values()
    assert "12 345,67 USD" in slots.values()


def test_guard_rejects_digits_and_unknown_tokens():
    slots = {"⟦MONTANT_1⟧": "12 345,67 USD", "⟦COMPTE_2⟧": "`0071:804ABCE90`"}
    ok, _ = guard("Le compte ⟦COMPTE_2⟧ a transféré 12 000 USD, ce qui est inhabituel et justifie un examen.", slots)
    assert not ok
    ok, _ = guard(
        "Le compte ⟦COMPTE_9⟧ a transféré ⟦MONTANT_1⟧, ce qui est inhabituel et justifie un examen approfondi.", slots
    )
    assert not ok
    ok, text = guard("<think>calcul</think>Le compte ⟦COMPTE_2⟧ a transféré ⟦MONTANT_1⟧, montant inhabituel.",
                     slots)  # fmt: skip
    assert ok and "12 345,67 USD" in text and "`0071:804ABCE90`" in text and "think" not in text
