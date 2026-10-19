"""How wrong the guarantee is, point by point along the Pareto front.

The front is not one answer, it is a menu: the operator picks the point that
fits their price cap. Each point carries its own reported guarantee, and each
was selected -- implicitly -- as the best of the candidates that scored near it
on one sample. So the question is not "how optimistic is the optimum" but

    at each point of the front, what is the probability that the guarantee we
    report is above the truth, and by how much?

Prediction worth refuting: where the front is FLAT, many schedules tie on their
true value and the sample decides which one wins, so the reported guarantee
should be more optimistic there. Where the front has slope, the winner wins on
merit and the gap should close.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_FRESH_POOLS = 8
FRESH_SEED_BASE = 700_000
POINTS_PER_FRONT = 10


def run_group(job: dict) -> list:
    """One (scale, day): build the fresh pools once, reuse for all its seeds."""
    import sys
    for e in job["sys_path"]:
        if e not in sys.path:
            sys.path.insert(0, e)
    import numpy as np
    from campaign import deep
    from simulator.fast_simulator_v2 import FastMCEvaluatorV2

    root = Path(job["root"])
    fresh = []
    for k in range(N_FRESH_POOLS):
        mc, mv = deep.build_day(root, job["day"], job["n_real"], job["scale"],
                                pool_seed=FRESH_SEED_BASE + k)
        n = len(mc)
        fresh.append(FastMCEvaluatorV2(
            mc, mv, deep.CHARGER_KW, deep.SOCKETS, deep.BESS_KWH, 0.7,
            np.tile(deep._hour_map(deep.PATTERN_ACTIVE), (n, 1)),
            np.tile(deep._prices(root), (n, 1)),
            grid_contract_power=deep.GRID_KW, boundary="cyclic"))

    rows = []
    for seed in job["seeds"]:
        cell = Path(job["cells"]) / f"d{job['day']}_x{seed}.json"
        if not cell.exists():
            continue
        front = json.loads(cell.read_text())["front"]
        if len(front) < 3:
            continue
        # Sample the front evenly by rank, so flat and steep stretches are both
        # represented rather than whichever one happens to hold more points.
        idx = np.unique(np.linspace(0, len(front) - 1,
                                    min(POINTS_PER_FRONT, len(front))).astype(int))
        costs = np.array([p["cost_p95"] for p in front])
        covs = np.array([p["er_p5"] for p in front])
        span_c = costs.max() - costs.min()
        for i in idx:
            p = front[i]
            out = np.array([ev.evaluate(p["schedule"])[0] for ev in fresh])
            # Local slope of the front: how much cost buys a point of coverage
            # here. Near zero means flat, and flat is where ties live.
            lo, hi = max(0, i - 2), min(len(front) - 1, i + 2)
            dcov = covs[hi] - covs[lo]
            slope = (costs[hi] - costs[lo]) / dcov if abs(dcov) > 1e-9 else np.nan
            rows.append({
                "scale": job["scale"], "day": job["day"], "search_seed": seed,
                "rank": int(i), "rank_frac": float(i / (len(front) - 1)),
                "in_sample": p["er_p5"], "cost": p["cost_p95"],
                "cost_frac": float((p["cost_p95"] - costs.min()) / span_c)
                if span_c > 1e-12 else 0.0,
                "out_of_sample": float(out.mean()),
                "optimism": float(p["er_p5"] - out.mean()),
                "p_overstated": float((out < p["er_p5"]).mean()),
                "estimator_sd": float(out.std(ddof=1)),
                "local_slope": float(slope) if slope == slope else float("nan"),
            })
    return rows
