"""The this study experiment: one day, one fleet, many arrival states.

The decision problem, and nothing else:

    A booking-based scheduler knows, the day before, WHICH vehicles are coming
    and therefore at what power each one charges. It does not know how much
    charge each one will turn up with. It must commit to a schedule anyway, so
    it optimises against the arrival states those vehicles could plausibly
    bring.

So one simulated day fixes a fleet and varies the arrival state:

    fleet        drawn once, from the fitted arrival counts. Fixed for the
                 whole Monte Carlo ensemble, because the booking revealed it.
    realisation  one draw of energy demanded per vehicle, from the fitted
                 energy distribution. The arrival SOC is derived from it, since
                 the two are the same quantity seen from either end.

Five days, each its own fleet, each an independent optimisation problem. They
are NOT compared against each other: different fleets are different problems.
They answer whether the method produces a usable front whatever fleet turns up.

Five search seeds per day separate search noise from sampling noise.

The result is read at FIXED COVERAGE LEVELS -- "what does it cost to reach 70%"
-- not at the front's endpoints. The cheap endpoint is flat, so which schedule
the optimiser stops on there is decided by the seed rather than by the problem.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# The station under study, fixed. Two sockets, 70 kW, no storage: storage is
# a separate question and mixing it in here would mean one experiment trying to
# settle two things at once.
SOCKETS = 2
CHARGER_KW = 70.0
BESS_KWH = 0.0
GRID_KW = None            # unconstrained: the binding limit is the sockets

# The four hourly patterns of the 3.0 TD calendar. P6 is the same overnight
# period in all of them, so the valley hours and the purchasable floor do not
# change between patterns: what changes is the price of the active periods, and
# therefore the cost of spilling out of the valley. The spread runs from
# 0.0237 EUR/kWh in winter to 0.0053 in mild, a factor of 4.5.
PATTERNS = {"winter": (1, 2), "shoulder": (2, 3),
            "medium": (3, 4), "mild": (4, 5)}
PATTERN = "winter"
PATTERN_ACTIVE = PATTERNS[PATTERN]

# Demand growth is more vehicles, not bigger ones: the fleet composition is
# multiplied. Scale 3 puts the station at roughly year 15-20 of the growth
# model. The TARIFF IS NOT TOUCHED, so demand is the only thing that moves and
# whatever changes can only be attributed to it.
DEMAND_SCALE = 1

N_DAYS = 5                # five fleets, five independent problems
N_REALISATIONS = 3000     # arrival states per fleet
SEARCH_SEEDS = (0, 1, 2, 3, 4)
N_EVAL = 30_000
POPULATION = 100

# Coverage levels the front is read at.
#
# The low levels look flat once the charging window is sized correctly: with
# the fleet fitting inside the 11 valley hours, everything up to 80% costs the
# purchasable floor. They stay because higher demand pushes the fleet out of
# the valley and the trade-off walks back down into them. The high levels are
# where the front lives at the demands simulated so far.
# 0.995 and not 1.00: a front whose best point covers everything reports
# er_p5 a few ulps below unity, so a level of exactly 1.00 is never
# attained and the column silently fills with NaN.
COVERAGE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995)

# Price budgets the front is read at, as a fraction above the purchasable
# floor. This is the reading an operator actually needs: not "what does 90%
# coverage cost" but "if I cap what I am willing to pay, how much coverage am
# I guaranteed on a bad day". Both readings are the same front; only one of
# them is a decision someone can act on.
PURCHASABLE_FLOOR = 0.178834          # EUR/kWh, 3.0 TD valley
COST_BUDGETS = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)

MIN_ENERGY_WH = 500.0     # the demand model's own floor


def _fitted_params(root: Path) -> dict:
    """Energy mean/std per charge type, as fitted to the measured sessions."""
    out = {}
    for code, name in (("S", "Slow"), ("N", "Normal"), ("F", "Fast")):
        doc = json.loads((root / "Data" / "output" /
                          f"distribution_params_{name}.json").read_text())
        out[code] = (doc["energy_stats"]["mean"], doc["energy_stats"]["std"])
    return out


def build_day(root: Path, day: int, n_real: int,
              scale: int = 1, pool_seed: int | None = None) -> tuple[list, list]:
    """One fleet, n_real arrival states for it.

    Returns (mc, maxv) in the shape the evaluator expects, where every entry of
    mc holds THE SAME vehicle ids and differs only in the energy each one asks
    for. maxv is then exact, so the decision vector has one variable per
    vehicle and the padding the calendar design needed disappears.
    """
    import sys
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from simulator.data_model.StatAnalisys2 import Simulation_stochastic_analysis

    sa = Simulation_stochastic_analysis()

    # The fleet: drawn once, from the fitted arrival counts. The day index is
    # the identity of the fleet, so day 3 is the same fleet on every rerun.
    rng_fleet = np.random.default_rng(20_000 + day)
    import random as _r
    state = _r.getstate(); np_state = np.random.get_state()
    _r.seed(20_000 + day); np.random.seed(20_000 + day)
    counts = {}
    for code, name in (("S", "Slow"), ("N", "Normal"), ("F", "Fast")):
        vehicle_series, _ = sa.generate_random_series(name, 8)
        counts[code] = int(vehicle_series[day % len(vehicle_series)])
    _r.setstate(state); np.random.set_state(np_state)

    counts = {c: k * scale for c, k in counts.items()}
    ids = [f"{code}{i}" for code in ("S", "N", "F")
           for i in range(1, counts[code] + 1)]
    if not ids:                       # a fleet of nobody is not a problem
        counts["N"] = 1
        ids = ["N1"]

    # The arrival states: one draw of energy per vehicle per realisation, from
    # the fitted distribution, truncated at the model's own floor.
    # `pool_seed` draws an INDEPENDENT set of arrival states for the SAME fleet.
    # It exists so that a schedule chosen on one sample can be scored on another,
    # and it goes through this function rather than a second generator so that
    # the vehicle insertion order is identical by construction. That order is
    # load-bearing: design_variables[i] addresses the i-th vehicle as inserted,
    # so a pool built elsewhere with a different order silently applies the
    # schedule to a permuted fleet. It cost me one wrong measurement to learn.
    params = _fitted_params(root)
    rng = np.random.default_rng(50_000 + day if pool_seed is None else pool_seed)
    mc = []
    for _ in range(n_real):
        mc.append({vid: float(max(MIN_ENERGY_WH,
                                  rng.normal(*params[vid[0]])))
                   for vid in ids})
    maxv = [counts["S"], counts["N"], counts["F"]]
    return mc, maxv


def _hour_map(active) -> np.ndarray:
    m = np.full(24, 6)
    m[9:14] = active[0]
    m[14:18] = active[1]
    m[18:22] = active[0]
    return m


def _prices(root: Path, fare="3.0 TD") -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(root / "Files" / "PreciosPeriodo.csv")
    df.columns = [c.strip() for c in df.columns]
    row = df[df["Tarifa"].str.strip() == fare].iloc[0]
    base = float(row["PreciokWh"])
    return np.array([base + float(row[f"Periodo{p}(kWh)"]) for p in range(1, 7)])


def cost_at_coverage(front: list, level: float) -> float:
    """Cheapest schedule on the front that reaches `level` coverage.

    NaN when the fleet cannot be served to that level at all: that is a
    property of the station against this fleet, and reporting it as a number
    would invent an answer the experiment does not have.
    """
    feasible = [p for p in front if p["er_p5"] >= level]
    return min((p["cost_p95"] for p in feasible), default=float("nan"))


def coverage_at_budget(front: list, budget: float) -> float:
    """Worst-case coverage guaranteed by the best schedule within a price cap.

    `budget` is a fraction above the purchasable floor. The coverage returned
    is the P5 one, so it is what the operator gets on a bad day, not on an
    average one: committing to a schedule means committing to its tail.

    NaN when no schedule on the front stays inside the cap, which means the cap
    is below what this fleet can be served for at all.
    """
    cap = PURCHASABLE_FLOOR * (1.0 + budget)
    reachable = [p["er_p5"] for p in front if p["cost_p95"] <= cap + 1e-12]
    return max(reachable, default=float("nan"))


def run_cell(job: dict) -> dict:
    import random
    import sys
    for entry in job["sys_path"]:
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import numpy as np
    from platypus import NSGAII, Problem, Real, nondominated
    from simulator.fast_simulator_v2 import FastMCEvaluatorV2

    root = Path(job["root"])
    t0 = time.perf_counter()

    mc, maxv = build_day(root, job["day"], job["n_real"],
                         job.get("scale", 1))
    n = len(mc)
    dpm = np.tile(_hour_map(PATTERNS[job.get("pattern", PATTERN)]), (n, 1))
    dpr = np.tile(_prices(root), (n, 1))
    charger_kw = float(job.get("charger_kw", CHARGER_KW))
    ev = FastMCEvaluatorV2(mc, maxv, charger_kw, job.get("sockets", SOCKETS),
                           float(job.get("bess_kwh", BESS_KWH)), 0.7,
                           dpm, dpr,
                           grid_contract_power=job.get("grid_kw", GRID_KW),
                           boundary="cyclic")

    n_vars = sum(maxv)

    # Pin the search. Platypus draws only from the stdlib `random`, so this
    # makes the whole run reproducible and keeps search noise separable from
    # the sampling noise carried by the realisations.
    random.seed(job["search_seed"])
    np.random.seed(job["search_seed"])

    # The quantile being optimised is an axis, because the selection bias of an
    # optimised quantile carries the density of the outcome at that quantile.
    # evaluate() returns the P5 of coverage; any other order has to be taken
    # from the raw realisations.
    quantile = float(job.get("quantile", 0.05))

    def f(x):
        sched = [int(v) for v in x]
        if abs(quantile - 0.05) < 1e-12:
            q_er, p95_c = ev.evaluate(sched)[:2]
        else:
            occ = np.zeros((ev.n_days, ev.total_max_veh, ev.n_slots), dtype=bool)
            ev._build_occupancy_one(sched, occ, 0)
            raw = ev._simulate(occ, ev.slot_prices, ev.charge_bess_flag,
                               ev.veh_energy_per_slot, ev.veh_total_demand,
                               return_raw=True)
            er = np.asarray(raw[0]).ravel(); ac = np.asarray(raw[1]).ravel()
            q_er = float(np.quantile(er, quantile))
            p95_c = float(np.quantile(ac, 0.95))
        return [-q_er, p95_c]

    prob = Problem(n_vars, 2)
    prob.types[:] = [Real(0, 95) for _ in range(n_vars)]
    prob.directions[:] = [Problem.MINIMIZE, Problem.MINIMIZE]
    prob.function = f
    alg = NSGAII(prob, population_size=job["population"])
    alg.run(job["n_eval"])

    front = [{"er_p5": -s.objectives[0], "cost_p95": s.objectives[1],
              "schedule": [int(v) for v in s.variables]}
             for s in sorted(nondominated(alg.result),
                             key=lambda s: s.objectives[1])]

    row = {
        "day": job["day"], "search_seed": job["search_seed"],
        "scale": job.get("scale", 1), "pattern": job.get("pattern", PATTERN),
        "bess_kwh": float(job.get("bess_kwh", BESS_KWH)),
        "sockets": job.get("sockets", SOCKETS),
        "charger_kw": charger_kw,
        "grid_kw": job.get("grid_kw", GRID_KW),
        "arm": job.get("arm", ""), "budget_eur": job.get("budget_eur", 0),
        "quantile": quantile,
        "n_vehicles": n_vars, "n_realisations": n,
        "fleet": "".join(f"{c}{k}" for c, k in
                         zip("SNF", maxv)),
        "front_points": len(front),
        "max_coverage": max((p["er_p5"] for p in front), default=float("nan")),
        "min_cost": min((p["cost_p95"] for p in front), default=float("nan")),
        "battery_truncation_vehicle_fraction":
            getattr(ev, "battery_truncation_vehicle_fraction", float("nan")),
        "demand_kwh_per_day_median":
            float(np.median(ev.veh_total_demand)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    for lvl in COVERAGE_LEVELS:
        row[f"cost_at_{int(lvl * 100)}"] = cost_at_coverage(front, lvl)
    for b in COST_BUDGETS:
        row[f"coverage_at_plus{int(b * 1000):03d}"] = coverage_at_budget(front, b)

    out = Path(job["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"row": row, "front": front}, indent=1))
    return row
