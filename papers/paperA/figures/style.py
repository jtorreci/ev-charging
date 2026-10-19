"""Shared look for every figure, and the rules that keep them honest.

Two conventions are enforced here rather than remembered per figure.

The purchasable floor is drawn on any axis that carries a unit cost. No
schedule can buy energy below the tariff's cheapest hour, so a line below that
floor is a defect and not a result; the previous corpus published several, and
a visible floor makes the next one impossible to miss.

Colours are chosen to survive greyscale printing and the common forms of colour
blindness, because a reviewer reading a printout is a reviewer we still have to
convince.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "run" / "data"
OUT = ROOT / "paper"

# Okabe-Ito, colour-blind safe and distinguishable in greyscale.
C = {
    "M1": "#D55E00", "M2": "#0072B2", "M3": "#009E73", "penalty": "#CC79A7",
    "winter": "#0072B2", "shoulder": "#56B4E9", "medium": "#E69F00",
    "mild": "#D55E00", "grey": "#666666", "floor": "#999999",
}
MARK = {"M1": "s", "M2": "o", "M3": "^", "penalty": "D"}
PATTERNS = ("winter", "shoulder", "medium", "mild")
PATTERN_LABEL = {"winter": "Winter [P1,P2,P6]", "shoulder": "Shoulder [P2,P3,P6]",
                 "medium": "Medium [P3,P4,P6]", "mild": "Mild [P4,P5,P6]"}

# The tariff's purchasable band, from PreciosPeriodo.csv: base energy term plus
# the cheapest and dearest period surcharges.
FLOOR = 0.178834
CEILING = 0.202574


def setup() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "lines.linewidth": 1.4, "lines.markersize": 4,
    })


def draw_floor(ax, label: bool = True, horizontal: bool = True) -> None:
    """Mark the purchasable floor on a unit-cost axis."""
    fn = ax.axhline if horizontal else ax.axvline
    fn(FLOOR, color=C["floor"], ls=":", lw=1.0, zorder=0,
       label="Purchasable floor" if label else None)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path
