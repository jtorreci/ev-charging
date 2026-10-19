"""The three figures of the manuscript, generated from data."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
FLOOR = 0.178834
VALLEY_KWH = 770.0
SCALES = ((1, "deep"), (3, "deep_x3"), (4, "deep_x4"))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.4,
})
MARK = {1: ("o", "#3b6ea5"), 3: ("s", "#c1621f"), 4: ("^", "#4f7a3a")}


def _rows(folder):
    return list(csv.DictReader(open(DATA / folder / "deep.csv")))


def fig_saturation():
    """Attainable coverage against the valley occupancy ratio."""
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for sc, folder in SCALES:
        rs = _rows(folder)
        xs, ys = [], []
        for d in sorted({int(r["day"]) for r in rs}):
            sub = [r for r in rs if int(r["day"]) == d]
            xs.append(float(sub[0]["demand_kwh_per_day_median"]) / VALLEY_KWH)
            ys.append(st.mean(float(r["max_coverage"]) for r in sub))
        m, c = MARK[sc]
        ax.scatter(xs, ys, marker=m, s=26, c=c, edgecolors="white",
                   linewidths=0.4, label=rf"$\times{sc}$ demand", zorder=3)
    ax.axvline(0.78, color="0.35", lw=0.8, ls="--", zorder=1)
    ax.annotate(r"$\rho^{*}\approx0.78$", xy=(0.78, 0.86), xytext=(1.02, 0.86),
                fontsize=7, color="0.25", va="center",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45"))
    ax.axhline(1.0, color="0.75", lw=0.6, ls=":", zorder=1)
    ax.set_xlabel(r"valley occupancy ratio $\rho$")
    ax.set_ylabel(r"attainable worst-case coverage  $\max P_5[\gamma]$")
    ax.set_ylim(0.70, 1.03)
    ax.set_xlim(0, 1.75)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(OUT / "fig_saturation.pdf")
    plt.close(fig)
    return "fig_saturation.pdf"


def fig_front():
    """One front, read along both of its axes."""
    cell = json.load(open(DATA / "deep_x4" / "cells" / "d0_x0.json"))
    front = sorted(cell["front"], key=lambda p: p["cost_p95"])
    cost = np.array([p["cost_p95"] for p in front])
    cov = np.array([p["er_p5"] for p in front])

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.step(cost, cov, where="post", color="0.25", lw=1.1, zorder=2)
    ax.scatter(cost, cov, s=7, c="0.25", zorder=3)

    # reading A: a coverage target, read down to its price
    tgt = 0.90
    i = int(np.argmax(cov >= tgt))
    ax.plot([cost.min(), cost[i]], [tgt, tgt], color="#c1621f", lw=0.9, ls="--", zorder=4)
    ax.plot([cost[i], cost[i]], [cov.min(), tgt], color="#c1621f", lw=0.9, ls="--", zorder=4)
    ax.scatter([cost[i]], [tgt], s=30, c="#c1621f", zorder=5)
    ax.annotate(f"target {tgt:.2f}\ncosts {cost[i]:.4f}", (cost[i], tgt),
                textcoords="offset points", xytext=(6, -22), fontsize=6.5,
                color="#c1621f")

    # reading B: a price cap, read across to what it guarantees
    cap = FLOOR * 1.02
    j = int(np.max(np.where(cost <= cap)))
    ax.plot([cap, cap], [cov.min(), cov[j]], color="#3b6ea5", lw=0.9, ls="-.", zorder=4)
    ax.plot([cost.min(), cap], [cov[j], cov[j]], color="#3b6ea5", lw=0.9, ls="-.", zorder=4)
    ax.scatter([cap], [cov[j]], s=30, c="#3b6ea5", zorder=5)
    ax.annotate(f"cap $+2\\%$\nguarantees {cov[j]:.3f}", (cap, cov[j]),
                textcoords="offset points", xytext=(-6, -20), fontsize=6.5,
                color="#3b6ea5", ha="right")

    ax.axvline(FLOOR, color="0.7", lw=0.6, ls=":", zorder=1)
    ax.text(FLOOR, cov.min(), " purchasable floor", fontsize=6.5, color="0.45",
            rotation=90, va="bottom")
    ax.set_xlabel(r"worst-case cost  $P_{95}[c]$  (EUR/kWh)")
    ax.set_ylabel(r"worst-case coverage  $P_5[\gamma]$")
    fig.savefig(OUT / "fig_front_readings.pdf")
    plt.close(fig)
    return "fig_front_readings.pdf"


def fig_selfdeception():
    """What the deterministic schedule expects against what it delivers."""
    rows = list(csv.DictReader(open(DATA / "baselines" / "baselines.csv")))
    det = [r for r in rows if r["arm"] == "deterministic" and r["scale"] in ("3", "4")]
    det.sort(key=lambda r: float(r["coverage_p5_best_at_floor"]) - float(r["coverage_p5"]))
    exp = [float(r["coverage_p5_best_at_floor"]) for r in det]
    got = [float(r["coverage_p5"]) for r in det]
    y = np.arange(len(det))

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for k, (e, g) in enumerate(zip(exp, got)):
        ax.plot([g, e], [k, k], color="0.7", lw=1.0, zorder=1)
    ax.scatter(exp, y, s=24, c="#c1621f", label="expected, on the mean instance",
               zorder=3)
    ax.scatter(got, y, s=24, c="#3b6ea5", label="obtained, against the real spread",
               zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([rf"$\times{r['scale']}$  {r['n_vehicles']} veh" for r in det],
                       fontsize=6.5)
    ax.set_xlabel(r"worst-case coverage  $P_5[\gamma]$")
    ax.set_xlim(0.5, 1.03)
    ax.set_ylim(-0.7, len(det) - 0.3)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "fig_selfdeception.pdf")
    plt.close(fig)
    return "fig_selfdeception.pdf"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_saturation, fig_front, fig_selfdeception):
        print(f"  wrote {fn()}")
