"""Schéma SVG d'un cas (sans dépendance) : comptes disposés en cercle, opérations numérotées dans l'ordre
chronologique avec leur montant. Couleurs par variables CSS, lisibles en thème clair comme sombre."""

from __future__ import annotations

import math
from html import escape


def case_svg(facts: dict, size: int = 520, max_nodes: int = 16) -> str:
    ops = sorted(facts["operations"], key=lambda o: o["index"])
    nodes: list[str] = []
    for o in ops:
        for a in (o["src"], o["dst"]):
            if a not in nodes:
                nodes.append(a)
    nodes = nodes[:max_nodes]
    if len(nodes) < 2:
        return ""
    cx = cy = size / 2
    r = size / 2 - 70
    pos = {
        a: (cx + r * math.cos(2 * math.pi * k / len(nodes) - math.pi / 2),
            cy + r * math.sin(2 * math.pi * k / len(nodes) - math.pi / 2))
        for k, a in enumerate(nodes)
    }  # fmt: skip
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" width="100%" '
        f'style="max-width:{size}px" role="img" aria-label="Schéma des flux du cas {escape(facts["case_id"])}">',
        "<defs><marker id='fl' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' "
        "orient='auto-start-reverse'><path d='M0,0L10,5L0,10z' fill='var(--accent,#8a3b12)'/></marker></defs>",
    ]
    for k, o in enumerate(ops, start=1):
        if o["src"] not in pos or o["dst"] not in pos:
            continue
        (x1, y1), (x2, y2) = pos[o["src"]], pos[o["dst"]]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1.0
        ux, uy = dx / d, dy / d
        sx, sy, ex, ey = x1 + 22 * ux, y1 + 22 * uy, x2 - 24 * ux, y2 - 24 * uy
        mx, my = (sx + ex) / 2 - 18 * uy, (sy + ey) / 2 + 18 * ux  # légère courbure : arêtes opposées distinctes
        parts.append(
            f"<path d='M{sx:.1f},{sy:.1f} Q{mx:.1f},{my:.1f} {ex:.1f},{ey:.1f}' fill='none' "
            f"stroke='var(--accent,#8a3b12)' stroke-width='1.6' marker-end='url(#fl)'/>"
            f"<text x='{mx:.1f}' y='{my:.1f}' font-size='11' text-anchor='middle' fill='var(--fg,#1d1d1f)'>"
            f"{k}. {escape(o['usd_fmt'])} $</text>"
        )
    for a, (x, y) in pos.items():
        label = a.split(":")[-1][-6:]
        parts.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='20' fill='var(--chip,#f1ece4)' stroke='var(--line,#e3e1db)'/>"
            f"<text x='{x:.1f}' y='{y + 4:.1f}' font-size='10' text-anchor='middle' fill='var(--fg,#1d1d1f)'>"
            f"…{escape(label)}</text><title>{escape(a)}</title>"
        )
    parts.append("</svg>")
    return "".join(parts)
