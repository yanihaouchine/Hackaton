"""Images pour les slides, à partir des résultats déjà calculés (aucun calcul de modèle ici).
- courbes d'entraînement du MLP retenu : loss (train, validation), accuracy (train, validation), PR-AUC de validation ;
- matrices de confusion MLP et forêt (validation ; test s'il a été évalué).
Usage (depuis neurones/) : ../.venv/bin/python scripts/plots.py
Sorties : results/figures/*.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nn.data import ARBRE, RESULTS

FIG = RESULTS / "figures"
# Palette de référence (validée : CVD ΔE 24,7, contraste ≥ 3:1 sur la surface claire)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAIN, VAL = "#2a78d6", "#eb6834"
STOP_FR = {"val_loss": "loss de validation", "val_pr_auc": "PR-AUC de validation"}
RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # bleu séquentiel, clair → foncé

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})  # fmt: skip


def fr(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def curves() -> None:
    t = json.loads((RESULTS / "train_mlp.json").read_text(encoding="utf-8"))
    run = t["seeds"][0]
    c = run["curves"]
    ep = [r["epoch"] for r in c]
    best = run["best_epoch"]
    forest_ap = json.loads((ARBRE / "artifacts/models/strict/model.json").read_text())["val_pr_auc"]
    prevalence = json.loads((RESULTS / "data_summary.json").read_text())["splits"]["val"]["rate"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        ("Loss (entropie croisée)", "train_loss", "val_loss"),
        ("Accuracy au seuil 0,5", "train_accuracy", "val_accuracy"),
    ]
    for ax, (title, ktr, kva) in zip(axes[:2], panels, strict=True):
        ax.plot(ep, [r[ktr] for r in c], color=TRAIN, marker="o", markersize=4, label="train")
        ax.plot(ep, [r[kva] for r in c], color=VAL, marker="o", markersize=4, label="validation")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
    ax = axes[1]
    ax.axhline(1 - prevalence, color=INK2, linestyle="--", linewidth=1.2)
    ax.annotate(f"« toujours licite » : {fr(100 * (1 - prevalence), 2)} %", (ep[-1], 1 - prevalence),
                xytext=(0, 6), textcoords="offset points", ha="right", color=INK2, fontsize=10)  # fmt: skip
    ax = axes[2]
    ax.plot(ep, [r["val_pr_auc"] for r in c], color=VAL, marker="o", markersize=4, label="MLP, validation")
    ax.axhline(forest_ap, color=INK2, linestyle="--", linewidth=1.2)
    ax.annotate(f"forêt de arbre/ : {fr(forest_ap)}", (ep[0], forest_ap), xytext=(0, 6), textcoords="offset points",
                color=INK2, fontsize=10)  # fmt: skip
    ax.set_title("PR-AUC de validation")
    ax.set_ylim(0, max(0.8, forest_ap + 0.1))
    ax.legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.axvline(best, color=INK2, linewidth=1, linestyle=":")
        ax.set_xlabel("epoch")
    axes[0].annotate(f"epoch retenue ({best})", (best, axes[0].get_ylim()[1]), xytext=(4, -14),
                     textcoords="offset points", color=INK2, fontsize=10)  # fmt: skip
    fig.suptitle(f"MLP : apprentissage (seed {run['seed']}, arrêt précoce sur la {STOP_FR[run['stop_on']]})",
                 x=0.01, ha="left", fontsize=13, fontweight="bold")  # fmt: skip
    fig.tight_layout()
    fig.savefig(FIG / "mlp_courbes.png", dpi=160)
    plt.close(fig)


def confusion_panel(ax, m: dict, title: str) -> None:
    """Couleur = part de la ligne (taux par classe réelle) ; texte = effectif et part."""
    cells = np.array([[m["VP"], m["FN"]], [m["FP"], m["VN"]]], dtype=float)
    rate = cells / cells.sum(axis=1, keepdims=True)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bleu", RAMP)
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    labels = [["VP", "FN"], ["FP", "VN"]]
    for i in range(2):
        for j in range(2):
            ink = SURFACE if rate[i, j] > 0.55 else INK
            n = f"{int(cells[i, j]):,}".replace(",", " ")
            ax.text(j, i, f"{labels[i][j]}\n{n}\n({fr(100 * rate[i, j], 1)} % de la ligne)", ha="center",
                    va="center", color=ink, fontsize=10.5)  # fmt: skip
    ax.set_xticks([0, 1], ["prédit illicite", "prédit licite"])
    ax.set_yticks([0, 1], ["illicite", "licite"])
    ax.set_ylabel("réalité")
    ax.set_title(title)
    ax.text(0.0, -0.22, f"précision {fr(m['precision'])} · rappel {fr(m['recall'])} · F1 {fr(m['f1'])}"
            f" · accuracy {fr(100 * m['accuracy'], 2)} %", transform=ax.transAxes, color=INK2, fontsize=10)  # fmt: skip


def confusions() -> None:
    conf = json.loads((RESULTS / "confusion.json").read_text(encoding="utf-8"))
    for split, label in (("val", "validation (07–08/09)"), ("test", "test (09–10/09)")):
        if split not in conf:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        confusion_panel(axes[0], conf[split]["mlp"], "MLP (neurones/)")
        confusion_panel(axes[1], conf[split]["forest"], "Forêt strict (arbre/)")
        fig.suptitle(f"Matrices de confusion, {label}, seuil fixé sur la validation", x=0.01, ha="left",
                     fontsize=13, fontweight="bold")  # fmt: skip
        fig.tight_layout()
        fig.savefig(FIG / f"confusion_{split}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    curves()
    confusions()
    print("\n".join(str(p.relative_to(RESULTS.parent)) for p in sorted(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
