"""The two tables of the manuscript and the summary numbers quoted in its text.

Everything here is read from `data/`; nothing is simulated. Run from the
artifact root:

    python3 tables/paper_a_tables.py > tables/paper_a_tables.out

Conventions, stated once because the paper relies on them:

* A cell is one (scale, day): one fleet. Each cell was searched under five
  seeds; a per-cell value is the mean over those seeds, skipping seeds for which
  the quantity does not exist (a coverage level the fleet cannot reach).
* rho is the median daily requested energy over the 770 kWh the winter valley
  can deliver (eq. rho in the methods).
* Where two cells share a composition, (a) is day 3 and (b) is day 0.
* Seed dispersion uses the population standard deviation over the five seeds.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLOOR = 0.178834          # purchasable floor, EUR/kWh
VALLEY_KWH = 770.0
RHO_STAR = 0.78
SCALES = ((1, "deep"), (3, "deep_x3"), (4, "deep_x4"))
PREMIUMS = (("+0%", "000"), ("+1%", "010"), ("+2%", "020"), ("+3%", "030"),
            ("+5%", "050"))
SEED_LEVELS = (50, 70, 90, 95, 99)
TWINS = {3: "(a)", 0: "(b)"}  # days that share a composition with another day
MACHINE_EPS = 1e-12


def _deep() -> pd.DataFrame:
    return pd.concat([pd.read_csv(DATA / f / "deep.csv") for _, f in SCALES],
                     ignore_index=True)


def _label(fleet: str, scale: int, day: int, cells: pd.DataFrame) -> str:
    parts = [f"{n}{c}" for c, n in re.findall(r"([SNF])(\d+)", fleet) if n != "0"]
    name = " ".join(parts)
    same = cells[(cells.scale == scale) & (cells.fleet == fleet)]
    return f"{name} {TWINS[day]}" if len(same) > 1 else name


def _cells(deep: pd.DataFrame) -> pd.DataFrame:
    g = deep.groupby(["scale", "day"])
    cells = g.agg(fleet=("fleet", "first"), veh=("n_vehicles", "first"),
                  demand=("demand_kwh_per_day_median", "first"),
                  max_cov=("max_coverage", "mean")).reset_index()
    cells["rho"] = cells.demand / VALLEY_KWH
    cells["label"] = [_label(r.fleet, r.scale, r.day, cells)
                      for r in cells.itertuples()]
    return cells


def table_valley(cells: pd.DataFrame) -> None:
    print("TABLE I  (tab:valley)  attainable coverage against valley occupancy")
    print(f"{'Scale':<6}{'Fleet':<12}{'Veh.':>5}{'Demand':>9}{'rho':>8}{'Max P5':>9}")
    for r in cells.sort_values("rho").itertuples():
        print(f"x{r.scale:<5}{r.label:<12}{r.veh:>5}{r.demand:>9.1f}"
              f"{r.rho:>8.3f}{r.max_cov:>9.4f}")
    print()


def table_cov_at_cost(deep: pd.DataFrame, cells: pd.DataFrame) -> None:
    print("TABLE II  (tab:cov_at_cost)  worst-case coverage under a price cap")
    cols = [f"coverage_at_plus{c}" for _, c in PREMIUMS]
    means = deep.groupby(["scale", "day"])[cols].mean()   # NaN-skipping
    print(f"{'Scale':<6}{'Fleet':<12}" + "".join(f"{p:>7}" for p, _ in PREMIUMS))
    order = (1, 4, 0, 3, 2)                                # the paper's row order
    for sc in (3, 4):
        for d in order:
            lab = cells[(cells.scale == sc) & (cells.day == d)].label.iloc[0]
            vals = means.loc[(sc, d)]
            print(f"x{sc:<5}{lab:<12}" + "".join(f"{v:>7.3f}" for v in vals))
    print()


def cost_of_targets(deep: pd.DataFrame, cells: pd.DataFrame) -> None:
    print("COST OF A COVERAGE TARGET  (sec:front, scale x4)")
    print("  premium over the floor, %, seed mean; '-' = level unreachable")
    for d in (3, 0, 2):
        sub = deep[(deep.scale == 4) & (deep.day == d)]
        lab = cells[(cells.scale == 4) & (cells.day == d)].label.iloc[0]
        txt = []
        for lv in (50, 70, 90, 95, 99):
            v = sub[f"cost_at_{lv}"]
            txt.append(f"{lv}:{(v.mean() / FLOOR - 1) * 100:.2f}"
                       if v.notna().all() else f"{lv}:-")
        print(f"  x4 {lab:<11}{'  '.join(txt)}   max P5 {sub.max_coverage.mean():.4f}")
    both = deep[(deep.scale == 4) & deep.day.isin([0, 3])]
    print("  x4 both 20-vehicle fleets pooled: "
          + "  ".join(f"{lv}:{(both[f'cost_at_{lv}'].mean() / FLOOR - 1) * 100:.2f}"
                      for lv in (70, 90, 95)))
    print(f"  optimisation time over {len(deep)} runs: median "
          f"{deep.seconds.median():.0f} s, max {deep.seconds.max():.0f} s")
    print()


def baselines() -> None:
    print("BOUNDARY SOLUTIONS  (sec:baselines_results, abstract)")
    b = pd.read_csv(DATA / "baselines" / "baselines.csv")
    cov = b.pivot_table(index=["scale", "day"], columns="arm", values="coverage_p5")
    margin = (cov.stochastic - cov.deterministic) * 100
    print(f"  stochastic >= deterministic in {int((margin >= -1e-9).sum())} "
          f"of {len(margin)} cells")
    for sc, m in margin.groupby(level="scale"):
        print(f"  mean margin x{sc}: {m.mean():.2f} pp")
    fa = b[b.arm == "free_arrival"]
    print(f"  free-arrival P95 cost, all cells: {fa.cost_p95.min():.4f} to "
          f"{fa.cost_p95.max():.4f} EUR/kWh "
          f"({(fa.cost_p95.min() / FLOOR - 1) * 100:.1f}% to "
          f"{(fa.cost_p95.max() / FLOOR - 1) * 100:.1f}% above the floor)")
    fa34 = fa[fa.scale.isin([3, 4])]
    print(f"  free-arrival P95 cost, x3 and x4: {fa34.cost_p95.min():.4f} to "
          f"{fa34.cost_p95.max():.4f} EUR/kWh")
    det = b[(b.arm == "deterministic") & b.scale.isin([3, 4])]
    gap = (det.coverage_p5_best_at_floor - det.coverage_p5) * 100
    breach = (det.cost_p95 / FLOOR - 1) * 100
    print(f"  deterministic self-overstatement, x3 and x4 (10 cells): "
          f"{gap.min():.2f} to {gap.max():.2f} pp")
    print(f"  deterministic price-cap breach, max: {breach.max():.2f}%")
    print()


def seed_stability(deep: pd.DataFrame, cells: pd.DataFrame) -> None:
    print("SEARCH-SEED STABILITY  (sec:stability)")
    print(f"  levels {SEED_LEVELS}; a reading counts when all five seeds reach "
          "the level; population SD")
    rho = cells.set_index(["scale", "day"]).rho
    sds, disp = [], []
    for (sc, d), sub in deep.groupby(["scale", "day"]):
        for lv in SEED_LEVELS:
            v = sub[f"cost_at_{lv}"].to_numpy()
            if np.isnan(v).any():
                continue
            sd = float(v.std(ddof=0))
            sds.append(sd)
            if sd > MACHINE_EPS:
                disp.append((sc, d, lv, sd, v.mean() - FLOOR, rho[(sc, d)]))
    sds = np.array(sds)
    spread = sds[sds > MACHINE_EPS]
    print(f"  readings {len(sds)}, seeds agree {int((sds <= MACHINE_EPS).sum())}, "
          f"disperse {len(spread)}")
    print(f"  dispersion SD: median {np.median(spread):.2e}, "
          f"max {spread.max():.2e} EUR/kWh")
    print(f"  dispersing readings in cells with rho >= {RHO_STAR}: "
          f"{sum(r >= RHO_STAR for *_, r in disp)} of {len(disp)}")
    print("    scale day level   rho     SD       signal   SD/signal")
    for sc, d, lv, sd, sig, r in disp:
        print(f"    x{sc:<4}{d:>3}{lv:>6}{r:>8.3f}{sd:>10.2e}{sig:>10.4f}"
              f"{sd / sig:>10.2f}")
    print()


def _t(x: np.ndarray, ddof: int) -> float:
    return float(x.mean() / (x.std(ddof=ddof) / np.sqrt(len(x))))


def optimism() -> None:
    print("OUT-OF-SAMPLE OPTIMISM  (sec:stability, 12 fresh ensembles)")
    o = pd.read_csv(DATA / "optimism" / "optimism.csv")
    for sc, sub in o.groupby("scale"):
        ch = sub[sub.role == "chosen"]
        print(f"  x{sc}: optimism {ch.optimism.mean():+.5f}, estimator SD "
              f"{ch.estimator_sd.mean():.5f}")
    pv = o.pivot_table(index=["scale", "day", "search_seed"], columns="role",
                       values="optimism")
    print("  selection contrast, chosen - control (paired over cells x seeds):")
    for sc, sub in pv.groupby(level="scale"):
        dlt = (sub.chosen - sub.control).to_numpy()
        print(f"    x{sc}: {dlt.mean():+.5f}  n={len(dlt)}  "
              f"t={_t(dlt, 0):.2f} (population SD)  t={_t(dlt, 1):.2f} (sample SD)")
    print()


def front_optimism() -> None:
    print("OPTIMISM ALONG THE FRONT  (sec:stability, 8 fresh ensembles)")
    f = pd.read_csv(DATA / "front_optimism" / "front_optimism.csv")
    print(f"  points sampled: {len(f)}")
    for sc, sub in f.groupby("scale"):
        print(f"  x{sc}: median P(guarantee above truth) {sub.p_overstated.median():.3f}")
    # Ten points per front at rank positions 0..9 (of 9); a third cannot hold an
    # equal share, so both ways of splitting 10 positions are shown.
    pos = (f.rank_frac * 9).round().astype(int)
    for name, edges in (("4-3-3", [-1, 3, 6, 9]), ("3-3-4", [-1, 2, 5, 9])):
        seg = pd.cut(pos, edges, labels=False)
        by_seg = f.groupby([f.scale, seg]).optimism.mean()
        print(f"  mean optimism per rank-third of the front ({name} split), max "
              f"over scales and thirds: {by_seg.max():.5f}")
    print("  flattest minus steepest third of local slope (NaN slopes dropped):")
    for sc in (3, 4):
        sub = f[(f.scale == sc)].dropna(subset=["local_slope"])
        r = sub.local_slope.rank(method="first")
        q = pd.qcut(r, 3, labels=False)
        a, b = sub[q == 0].optimism, sub[q == 2].optimism
        t = stats.ttest_ind(a, b, equal_var=False).statistic
        print(f"    x{sc} pooled thirds:    {a.mean() - b.mean():+.5f}  (t={t:+.2f}, "
              f"Welch, n={len(a)}/{len(b)})")
        q2 = sub.groupby(["day", "search_seed"]).local_slope.transform(
            lambda z: pd.qcut(z.rank(method="first"), 3, labels=False)
            if len(z) >= 3 else pd.Series(np.nan, index=z.index))
        a, b = sub[q2 == 0].optimism, sub[q2 == 2].optimism
        t = stats.ttest_ind(a, b, equal_var=True).statistic
        print(f"    x{sc} per-front thirds: {a.mean() - b.mean():+.5f}  (t={t:+.2f}, "
              f"Student, n={len(a)}/{len(b)})")
    print("  note: the manuscript quotes -0.00008 (t=-0.36) at x3 and +0.00031 "
          "(t=+1.30) at x4;\n  the script that produced them is not preserved, "
          "and neither split above\n  reproduces both pairs exactly.")
    print()


if __name__ == "__main__":
    deep = _deep()
    cells = _cells(deep)
    table_valley(cells)
    table_cov_at_cost(deep, cells)
    cost_of_targets(deep, cells)
    baselines()
    seed_stability(deep, cells)
    optimism()
    front_optimism()
