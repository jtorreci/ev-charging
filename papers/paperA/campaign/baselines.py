"""The two boundary solutions the stochastic schedule has to beat.

Neither is a rival formulation of our method -- that comparison was dropped as
not being the object of the paper. These are controls: they establish that the
method is needed at all.

    free arrival   nobody schedules. Vehicles turn up spread over the day.
                   This is the floor: what the station does with no algorithm.

    deterministic  the schedule is optimised against ONE instance in which every
                   vehicle asks for its type's MEAN energy, and is then made to
                   face the real spread of arrival states. This is the question
                   a reviewer will ask -- why not just optimise the expected
                   demand? -- answered with a number instead of an argument.

Both are scored on the SAME arrival states as the stochastic schedule, so the
only thing that differs is what the schedule was chosen against.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_FREE_DRAWS = 50


def _mean_pool(mc, params):
    """One instance: every vehicle asks its type's mean. Keys and order kept
    from a real pool, because the insertion order addresses the schedule."""
    return [{vid: float(params[vid[0]][0]) for vid in mc[0].keys()}]


def run_group(job: dict) -> list:
    import sys
    for e in job["sys_path"]:
        if e not in sys.path:
            sys.path.insert(0, e)
    import random
    import numpy as np
    from platypus import NSGAII, Problem, Real, nondominated
    from campaign import deep
    from simulator.fast_simulator_v2 import FastMCEvaluatorV2

    root = Path(job["root"])
    scale, day = job["scale"], job["day"]
    mc, maxv = deep.build_day(root, day, job["n_real"], scale)
    params = deep._fitted_params(root)
    n_vars = sum(maxv)

    def make(pool):
        n = len(pool)
        return FastMCEvaluatorV2(
            pool, maxv, deep.CHARGER_KW, deep.SOCKETS, deep.BESS_KWH, 0.7,
            np.tile(deep._hour_map(deep.PATTERN_ACTIVE), (n, 1)),
            np.tile(deep._prices(root), (n, 1)),
            grid_contract_power=deep.GRID_KW, boundary="cyclic")

    ev_true = make(mc)                      # the real spread: the judge
    ev_mean = make(_mean_pool(mc, params))  # the deterministic instance
    cap = deep.PURCHASABLE_FLOOR
    rows = []

    # -- free arrival ------------------------------------------------------
    rng = np.random.default_rng(11_000 + 100 * scale + day)
    best = None
    for _ in range(N_FREE_DRAWS):
        sched = [int(v) for v in rng.integers(0, 96, size=n_vars)]
        p5, c95 = ev_true.evaluate(sched)[:2]
        if c95 <= cap + 1e-12 and (best is None or p5 > best[0]):
            best = (p5, c95)
    draws = np.array([ev_true.evaluate([int(v) for v in rng.integers(0, 96, size=n_vars)])[:2]
                      for _ in range(N_FREE_DRAWS)])
    rows.append({"scale": scale, "day": day, "arm": "free_arrival",
                 "n_vehicles": n_vars,
                 "coverage_p5": float(draws[:, 0].mean()),
                 "cost_p95": float(draws[:, 1].mean()),
                 "coverage_p5_best_at_floor": float(best[0]) if best else float("nan")})

    # -- deterministic -----------------------------------------------------
    random.seed(0); np.random.seed(0)

    def f(x):
        p5, c95 = ev_mean.evaluate([int(v) for v in x])[:2]
        return [-p5, c95]

    prob = Problem(n_vars, 2)
    prob.types[:] = [Real(0, 95) for _ in range(n_vars)]
    prob.directions[:] = [Problem.MINIMIZE, Problem.MINIMIZE]
    prob.function = f
    alg = NSGAII(prob, population_size=deep.POPULATION)
    alg.run(job["n_eval"])
    det_front = sorted(nondominated(alg.result), key=lambda s: s.objectives[1])
    # The schedule a planner would ship: cheapest price, best coverage there --
    # chosen on the deterministic instance, exactly as if the spread were unknown.
    at_floor = [s for s in det_front if s.objectives[1] <= cap + 1e-12] or det_front
    pick = max(at_floor, key=lambda s: -s.objectives[0])
    sched = [int(v) for v in pick.variables]
    true_p5, true_c95 = ev_true.evaluate(sched)[:2]
    rows.append({"scale": scale, "day": day, "arm": "deterministic",
                 "n_vehicles": n_vars,
                 "coverage_p5": float(true_p5), "cost_p95": float(true_c95),
                 "coverage_p5_best_at_floor": float(-pick.objectives[0])})

    # -- stochastic, read off the stored front -----------------------------
    #
    # At the SAME budget the deterministic schedule actually ends up spending,
    # not at the floor. Scored against the real spread, the deterministic
    # schedule overshoots its price cap -- by 5.5% in the worst cell -- because
    # it was chosen without knowing a spread existed. Comparing its coverage
    # against a stochastic arm pinned to the floor would let it buy coverage
    # with money the other one was forbidden to spend, and the two arms would
    # differ in two respects instead of one.
    budget = max(cap, float(true_c95))
    cells = Path(job["cells"])
    at_floor_vals, at_budget_vals = [], []
    for s in job["seeds"]:
        c = cells / f"d{day}_x{s}.json"
        if not c.exists():
            continue
        fr = json.loads(c.read_text())["front"]
        ok = [p["er_p5"] for p in fr if p["cost_p95"] <= cap + 1e-12]
        if ok:
            at_floor_vals.append(max(ok))
        ok2 = [p["er_p5"] for p in fr if p["cost_p95"] <= budget + 1e-12]
        if ok2:
            at_budget_vals.append(max(ok2))
    if at_budget_vals:
        rows.append({"scale": scale, "day": day, "arm": "stochastic",
                     "n_vehicles": n_vars,
                     "coverage_p5": float(np.mean(at_budget_vals)),
                     "cost_p95": budget,
                     "coverage_p5_best_at_floor":
                         float(np.mean(at_floor_vals)) if at_floor_vals else float("nan")})
    return rows
