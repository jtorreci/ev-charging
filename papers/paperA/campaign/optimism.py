"""How much the reported guarantee overstates the truth.

The schedule is chosen by scoring ~30,000 candidates against ONE sample of
3,000 arrival states. Each individual score is a percentile over those states
and is essentially unbiased. The selection is not: the maximum over many noisy
scores favours whichever candidate the sample happened to flatter, so the
winner's reported guarantee is optimistic.

This measures that gap the only way it can be measured -- by scoring the chosen
schedule against arrival states it was not chosen on -- and contrasts it with a
schedule from the same front that was NOT chosen. Without that contrast the
number is just estimator noise wearing a hat.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_FRESH_POOLS = 12
FRESH_SEED_BASE = 900_000


def _evaluators(root, day, scale, n_real, maxv, seeds, deep):
    from simulator.fast_simulator_v2 import FastMCEvaluatorV2
    out = []
    for sd in seeds:
        mc, mv = deep.build_day(root, day, n_real, scale, pool_seed=sd)
        n = len(mc)
        out.append(FastMCEvaluatorV2(
            mc, mv, deep.CHARGER_KW, deep.SOCKETS, deep.BESS_KWH, 0.7,
            np.tile(deep._hour_map(deep.PATTERN_ACTIVE), (n, 1)),
            np.tile(deep._prices(root), (n, 1)),
            grid_contract_power=deep.GRID_KW, boundary="cyclic"))
    return out


def run_cell(job: dict) -> list:
    import sys
    for e in job["sys_path"]:
        if e not in sys.path:
            sys.path.insert(0, e)
    from campaign import deep

    root = Path(job["root"])
    cell = json.loads(Path(job["cell"]).read_text())
    front, row = cell["front"], cell["row"]
    if not front:
        return []

    _, maxv = deep.build_day(root, job["day"], 8, job["scale"])
    fresh = _evaluators(root, job["day"], job["scale"], job["n_real"], maxv,
                        [FRESH_SEED_BASE + k for k in range(N_FRESH_POOLS)],
                        deep)

    cap = deep.PURCHASABLE_FLOOR
    at_floor = [p for p in front if p["cost_p95"] <= cap + 1e-12]
    if not at_floor:
        at_floor = front
    chosen = max(at_floor, key=lambda p: p["er_p5"])
    # The contrast: the same front, a schedule nobody selected. Its in-sample
    # score was not the argmax, so it carries estimator noise but no winner's
    # curse. The difference between the two is the part selection is to blame for.
    others = [p for p in front if p is not chosen]
    control = others[len(others) // 2] if others else None

    rows = []
    for label, point in (("chosen", chosen), ("control", control)):
        if point is None:
            continue
        out = np.array([ev.evaluate(point["schedule"])[0] for ev in fresh])
        rows.append({
            "scale": job["scale"], "day": job["day"],
            "search_seed": job["search_seed"], "role": label,
            "n_vehicles": row["n_vehicles"], "n_real": job["n_real"],
            "in_sample": point["er_p5"],
            "out_of_sample": float(out.mean()),
            "optimism": float(point["er_p5"] - out.mean()),
            "estimator_sd": float(out.std(ddof=1)),
            "optimism_in_sd": float((point["er_p5"] - out.mean()) / out.std(ddof=1))
            if out.std(ddof=1) > 0 else float("nan"),
        })
    return rows
