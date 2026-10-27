"""Images pour les slides, à partir des résultats déjà calculés (aucun calcul de modèle ici).

Repris de la v1 de neurones/ (commit 0c12204), même palette :
- courbes d'entraînement du MLP livré : perte d'entraînement, PR-AUC de validation (critère d'arrêt précoce) ;
- matrices de confusion MLP et forêt (validation ; test s'il a été évalué).
Usage (depuis neurones/) : python -m mlp.plots [--model h64-32_n100_s42]  ->  results/figures/*.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlp.common import REPO, RESULTS

FIG = RESULTS / "figures"
# Palette de référence (validée : CVD ΔE 24,7, contraste ≥ 3:1 sur la surface claire)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAIN, VAL = "#2a78d6", "#eb6834"
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


def curves(model: str) -> None:
    run = json.loads((RESULTS / f"val_mlp_{model}.json").read_text(encoding="utf-8"))
    c = run["history"]
    ep = [r["epoch"] for r in c]
    best = run["best_epoch"]
    forest_ap = json.loads((REPO / "arbre/artifacts/models/strict/model.json").read_text())["val_pr_auc"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(ep, [r["train_loss"] for r in c], color=TRAIN, marker="o", markersize=4, label="train")
    ax.set_title("Perte d'entraînement (entropie croisée)")
    ax.legend(frameon=False, loc="best")
    ax = axes[1]
    ax.plot(ep, [r["val_pr_auc"] for r in c], color=VAL, marker="o", markersize=4, label="MLP, validation")
    ax.axhline(forest_ap, color=INK2, linestyle="--", linewidth=1.2)
    ax.annotate(f"forêt de arbre/ : {fr(forest_ap)}", (ep[0], forest_ap), xytext=(0, 6), textcoords="offset points",
                color=INK2, fontsize=10)  # fmt: skip
    ax.set_title("PR-AUC de validation")
    ax.set_ylim(0, max(0.8, forest_ap + 0.1))
    ax.legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.axvline(best, color=INK2, linewidth=1, linestyle=":")
        ax.set_xlabel("époque")
    axes[0].annotate(f"époque retenue ({best})", (best, axes[0].get_ylim()[1]), xytext=(4, -14),
                     textcoords="offset points", color=INK2, fontsize=10)  # fmt: skip
    seed = run["run"]["seed"]
    fig.suptitle(f"MLP {model.split('_')[0][1:].replace('-', ' → ')} : apprentissage (seed {seed}, arrêt précoce "
                 "sur la PR-AUC de validation)", x=0.01, ha="left", fontsize=13, fontweight="bold")  # fmt: skip
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
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="h64-32_n100_s42")
    a = p.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    curves(a.model)
    confusions()
    print("\n".join(str(p.relative_to(RESULTS.parent)) for p in sorted(FIG.glob("*.png"))))


if __name__ == "__main__":
    main()
