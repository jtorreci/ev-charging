"""
fast_simulator_v2.py -- Vectorized MC evaluator, v2
===================================================
Extends FastMCEvaluator with:
  1. PER-DAY tariff: each MC day carries its own 24h period map + prices
     (enables full-year / stratified-by-pattern designs).
  2. evaluate_batch(X): evaluate a whole POPULATION of schedules by folding the
     population into the leading "day" axis (pop*n_days), so the 96-slot
     simulation runs once over all (schedule, day) pairs.

Parity: with a uniform (single-month) tariff, evaluate(x) reproduces v1 exactly.
"""
import numpy as np
import hashlib
import random as _random
from pathlib import Path

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

    def njit(*a, **k):
        def _wrap(f):
            return f
        return _wrap if (a and callable(a[0]) is False) else (a[0] if a else _wrap)


@njit(cache=True, inline="always")
def _interp1(x, xs, ys):
    """Scalar np.interp: clamp outside the support, linear in between."""
    n = xs.shape[0]
    if x <= xs[0]:
        return ys[0]
    if x >= xs[n - 1]:
        return ys[n - 1]
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    dx = xs[hi] - xs[lo]
    if dx == 0.0:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / dx


@njit(cache=True)
def _percentile_linear(sorted_a, q):
    """numpy's default percentile: linear interpolation between order statistics."""
    n = sorted_a.shape[0]
    if n == 1:
        return sorted_a[0]
    pos = (q / 100.0) * (n - 1)
    lo = int(pos)
    if lo >= n - 1:
        return sorted_a[n - 1]
    frac = pos - lo
    return sorted_a[lo] + (sorted_a[lo + 1] - sorted_a[lo]) * frac


@njit(cache=True)
def _stats_numba(er, ac):
    """(P5 of er, P95 of ac, mean er, std er, mean ac, std ac), one pass each."""
    n = er.shape[0]
    se = 0.0
    sa = 0.0
    for i in range(n):
        se += er[i]
        sa += ac[i]
    me = se / n
    ma = sa / n
    ve = 0.0
    va = 0.0
    for i in range(n):
        de = er[i] - me
        da = ac[i] - ma
        ve += de * de
        va += da * da
    er_sorted = np.sort(er)
    ac_sorted = np.sort(ac)
    return (_percentile_linear(er_sorted, 5.0),
            _percentile_linear(ac_sorted, 95.0),
            me, np.sqrt(ve / n), ma, np.sqrt(va / n))


@njit(cache=True)
def _slot_loop_numba(occ, prices, cflags, veh_eps, soc_open, cep, grid_budget,
                     use_grid, has_bess, bess_cap, slots_per_hour,
                     hours_per_slot, charge_soc, charge_energy, charge_budget):
    """The 96-slot simulation, day by day.

    Same arithmetic as the numpy path, but the per-slot work is scalar, so the
    ~20 array operations per slot (each a separate numpy dispatch on a small
    array) collapse into registers. The day loop is outermost so the state of
    charge stays in a register across the day.
    """
    D = occ.shape[0]
    V = occ.shape[1]
    S = occ.shape[2]
    total_cost = np.zeros(D)
    total_served = np.zeros(D)
    peak_grid = np.zeros(D)
    perslot = np.zeros((D, S))
    soc_close = np.zeros(D)
    refill = np.zeros(D)
    purchased = np.zeros(D)

    for d in range(D):
        soc = soc_open[d] if has_bess else 0.0
        best_charge_price = np.inf
        best_any_price = np.inf
        cost = 0.0
        served = 0.0
        peak = 0.0
        bought = 0.0
        taken = 0.0          # energy stored so far today, against the budget
        for s in range(S):
            price = prices[d, s]
            if price < best_any_price:
                best_any_price = price
            if cflags[d, s] and price < best_charge_price:
                best_charge_price = price
            td = 0.0
            for v in range(V):
                if occ[d, v, s]:
                    td += veh_eps[d, v]

            if not has_bess:
                available = cep
                if use_grid and grid_budget < available:
                    available = grid_budget
                ratio = 1.0
                if td > available:
                    ratio = available / (td if td > 1e-10 else 1e-10)
                es = ratio * td
                cost += es * price
                served += es
                bought += es
                perslot[d, s] = es
                g = es / hours_per_slot
                if g > peak:
                    peak = g
                continue

            charging = cflags[d, s]
            bess_charge = 0.0
            if charging and soc < 1.0:
                want = _interp1(soc, charge_soc, charge_energy)
                # Never buy more than fits: clamping the state of charge after
                # billing would pay for energy that then vanishes, which shows
                # up as a small positive residual in the energy balance.
                headroom = (1.0 - soc) * bess_cap
                if want > headroom:
                    want = headroom
                # Do not store more than the day is expected to need outside the
                # cheap window. The budget is a forecast from the committed
                # schedule and the demand distribution, both of which the
                # operator holds; it is not the realised figure, which would
                # give the battery foresight it cannot have.
                left = charge_budget - taken
                if left < 0.0:
                    left = 0.0
                if want > left:
                    want = left
                if use_grid:
                    veh_need = td if td < cep else cep
                    room = grid_budget - veh_need
                    if room < 0.0:
                        room = 0.0
                    bess_charge = want if want < room else room
                else:
                    bess_charge = want
            soc = soc + bess_charge / bess_cap
            if soc > 1.0:      # numerical guard only; headroom was applied above
                soc = 1.0
            taken += bess_charge
            cost += bess_charge * price

            bess_avail = soc * bess_cap / slots_per_hour
            if charging:
                available = cep
                if use_grid:
                    room = grid_budget - bess_charge
                    if room < 0.0:
                        room = 0.0
                    if room < available:
                        available = room
            else:
                grid_energy = cep
                if use_grid and grid_budget < grid_energy:
                    grid_energy = grid_budget
                available = grid_energy + bess_avail

            ratio = 1.0
            if td > available:
                ratio = available / (td if td > 1e-10 else 1e-10)
            es = ratio * td

            bd = 0.0
            if not charging:
                bd = bess_avail if bess_avail < es else es
                soc = soc - bd / bess_cap
                if soc < 0.0:
                    soc = 0.0
            cost += (es - bd) * price

            served += es
            grid_energy = es - bd + bess_charge
            bought += grid_energy
            perslot[d, s] = es
            g = grid_energy / hours_per_slot
            if g > peak:
                peak = g

        total_cost[d] = cost
        total_served[d] = served
        peak_grid[d] = peak
        soc_close[d] = soc
        refill[d] = best_charge_price if np.isfinite(best_charge_price) else best_any_price
        purchased[d] = bought

    return (total_cost, total_served, peak_grid, perslot, soc_close, refill,
            purchased)


@njit(cache=True)
def _build_occ_numba(arrival_slots, veh_count, veh_slots_needed, socket_limit, n_slots, occ, base):
    """Greedy socket-contention occupancy fill, JIT-compiled. Bit-identical to the
    Python loop: vehicles processed in (arrival slot, then index) order; each takes
    a contiguous run from its start until it fills its need or hits a full slot."""
    n_days = veh_count.shape[0]
    for d in range(n_days):
        nv = veh_count[d]
        if nv <= 0:
            continue
        # stable order by arrival slot (ties -> original index): unique key
        key = np.empty(nv, dtype=np.int64)
        for i in range(nv):
            key[i] = arrival_slots[i] * 1000 + i
        order = np.argsort(key)
        slot_count = np.zeros(n_slots, dtype=np.int64)
        for oi in range(nv):
            v = order[oi]
            start = arrival_slots[v]
            if start > n_slots - 1:
                start = n_slots - 1
            end = start + veh_slots_needed[d, v]
            if end > n_slots:
                end = n_slots
            for s in range(start, end):
                if slot_count[s] < socket_limit:
                    occ[base + d, v, s] = True
                    slot_count[s] += 1
                else:
                    break


# The demand pool stores energies in Wh (Data/StatAnalisys2.py draws them with
# MIN_ENERGY_WH = 500), while battery capacities are in kWh. Dividing one by the
# other without converting made every energy/capacity ratio ~526, so
# min(0.9, soc + ratio) pinned the target SOC at 0.9 for 100% of vehicles and
# the fitted demand distribution never reached the simulator at all. The unit
# belongs in the name so the next reader sees the mismatch at the call site.
WH_PER_KWH = 1000.0

# Vehicles are charged towards this state. It is the operator's target, not a
# physical limit; the physical limit is the battery, applied below.
TARGET_SOC = 0.9


def draw_fleet(day_data):
    """Draw the fixed attributes of the booked fleet.

    A booking-based scheduler learns the day before which vehicles are coming
    and of what type, and the type fixes the charging power while the vehicle
    fixes its battery capacity. So this stream is seeded by the COMPOSITION
    alone and draws identically for every Monte Carlo realisation of the same
    fleet.

    It used to be seeded by md5(sorted(day_data.items())), and day_data maps
    vehicle -> energy, so one vehicle's demand moved every other vehicle's
    power and capacity: shifting S1's energy by 200 Wh took F1's power from 75
    to 131 kW. The fleet was resampled by the very quantity the Monte Carlo
    varies, which is the opposite of what a booking reveals.

    :param day_data: dict vehicle_id -> energy demanded, in Wh.
    :return: dict vehicle_id -> (max_charge_power_kw, battery_capacity_kwh).
    """
    ids = sorted(day_data.keys())
    seed = int(hashlib.md5("|".join(ids).encode()).hexdigest(), 16) % (2 ** 32)

    # Save and restore the RNG state so the NSGA-II operators are untouched.
    state = _random.getstate()
    _random.seed(seed)
    params = {}
    for vid in ids:
        ct = vid[0]
        power = (_random.uniform(50, 150) if ct == 'F'
                 else _random.uniform(10, 50) if ct == 'N'
                 else _random.uniform(5, 10))
        params[vid] = (power, _random.uniform(50, 100))
    _random.setstate(state)
    return params


def build_day_vehicles(day_data):
    """Turn one Monte Carlo realisation into the vehicles it describes.

    Arrival SOC and energy demanded are two views of the same quantity, so only
    one of them may be drawn. The fitted distribution is on ENERGY, measured
    from real charging sessions, so energy is drawn and the SOC is derived:

        soc_initial = max(0, TARGET_SOC - energy_kwh / capacity_kwh)

    Drawing both, as the code did before, over-determined the demand and let a
    clamp reconcile them silently.

    The max(0, ...) is the battery itself: a vehicle asking for more than its
    pack holds gets what fits. That truncation is physical and it is measured
    (about 9% of vehicles, about 2% of energy), not an artefact to hide.

    :param day_data: dict vehicle_id -> energy demanded, in Wh.
    :return: list of (vehicle_id, power_kw, capacity_kwh, soc_initial,
        soc_target, energy_to_charge_kwh), in day_data's OWN order. The order
        matters: design_variables[i] addresses the i-th vehicle as inserted,
        and sorting here would silently remap the decision vector.
    """
    params = draw_fleet(day_data)
    out = []
    for vid, energy_wh in day_data.items():
        power_kw, capacity_kwh = params[vid]
        energy_kwh = energy_wh / WH_PER_KWH
        soc_initial = max(0.0, TARGET_SOC - energy_kwh / capacity_kwh)
        energy_to_charge_kwh = capacity_kwh * (TARGET_SOC - soc_initial)
        out.append((vid, power_kw, capacity_kwh, soc_initial,
                    TARGET_SOC, energy_to_charge_kwh))
    return out


class FastMCEvaluatorV2:
    def __init__(self, mc_data, max_vehicles, charger_power_kw, socket_limit,
                 bess_capacity_kwh, bess_initial_soc,
                 day_period_map, day_prices,
                 grid_contract_power=None, slot_duration_min=15,
                 boundary="cyclic", settle_storage=None,
                 day_hourly_prices=None):
        """
        day_period_map : (n_days, 24) int  -- tariff period (1-6) per hour, per day
        day_prices     : (n_days, 6)  float -- price per period (index period-1), per day
        (A single (24,) map / (6,) prices is broadcast to all days for parity.)

        day_hourly_prices : (n_days, 24) float, optional -- the price of each
        hour directly, overriding the per-period lookup. Some tariff scenarios
        act on the RATE rather than on the regulated surcharge (a super-valley
        discount, a stochastic wholesale draw), and those cannot be expressed
        as a per-period price at all: two hours of the same period can cost
        different amounts. The period map is still required, because it is what
        decides when the battery charges.
        """
        self.n_slots = 24 * (60 // slot_duration_min)
        self.slot_duration_min = slot_duration_min
        self.hours_per_slot = slot_duration_min / 60.0
        self.slots_per_hour = 60 // slot_duration_min
        self.charger_power_kw = charger_power_kw
        self.charger_energy_per_slot = charger_power_kw * self.hours_per_slot
        self.socket_limit = socket_limit
        # Boundary design for the battery's state of charge. A simulated day is
        # an independent Monte Carlo draw, so what happens at the day boundary
        # is a modelling choice and not a detail:
        #   "open"       legacy. Every day opens at bess_initial_soc and closes
        #                wherever it lands, so stored energy is not conserved:
        #                a day ending low serves energy that was never bought,
        #                a day ending high buys energy that is then discarded.
        #                Kept only to reproduce previously published numbers.
        #   "cyclic"     the day is closed back to the SOC it opened at, priced
        #                at that day's cheapest charge period. Days stay
        #                exchangeable and no energy is created or destroyed.
        #   "stationary" the opening SOC is itself a random variable whose law
        #                is a fixed point of the daily map (see
        #                solve_stationary_soc). No settlement is applied: the
        #                balance closes in distribution rather than per day.
        #   "chain"      the reference. Days are walked in calendar order and
        #                each opens at the previous day's closing SOC, so the
        #                balance closes exactly and nothing is assumed about
        #                the boundary. Costs a sequential pass over days (see
        #                simulate_chain), so it is used to validate the cheaper
        #                designs rather than inside the optimiser.
        if settle_storage is not None:      # backwards-compatible alias
            boundary = "cyclic" if settle_storage else "open"
        if boundary not in ("open", "cyclic", "stationary", "chain"):
            raise ValueError(f"unknown boundary design: {boundary!r}")
        self.boundary = boundary
        self.has_bess = bess_capacity_kwh > 0
        self.bess_capacity = bess_capacity_kwh
        self.bess_initial_soc = bess_initial_soc
        # Per-day opening SOC, set by solve_stationary_soc under the stationary
        # design. None means "use the scalar for every day".
        self.soc_open = None
        # Convergence record of the last solve_stationary_soc call. A caller
        # that prices a stationary result must check `stationary_converged`;
        # `stationary_residual` is the unpriced net draw in SOC units.
        self.stationary_ks = 0.0
        self.stationary_residual = 0.0
        self.stationary_converged = True
        self.grid_contract_power = grid_contract_power
        self.grid_budget = (grid_contract_power * self.hours_per_slot
                            if grid_contract_power is not None else None)

        self._precompute_vehicles(mc_data, max_vehicles)
        n_days = self.n_days

        # Broadcast single-month tariff to per-day if needed
        dpm = np.asarray(day_period_map)
        if dpm.ndim == 1:
            dpm = np.tile(dpm, (n_days, 1))
        dpr = np.asarray(day_prices, dtype=float)
        if dpr.ndim == 1:
            dpr = np.tile(dpr, (n_days, 1))

        dhp = None
        if day_hourly_prices is not None:
            dhp = np.asarray(day_hourly_prices, dtype=float)
            if dhp.ndim == 1:
                dhp = np.tile(dhp, (n_days, 1))
            if dhp.shape != (n_days, 24):
                raise ValueError(
                    f"day_hourly_prices must be (n_days, 24), got {dhp.shape}")

        # Per-day slot prices (n_days, n_slots) and BESS-charge flags
        self.slot_prices = np.zeros((n_days, self.n_slots))
        self.charge_bess_flag = np.zeros((n_days, self.n_slots), dtype=bool)
        # Two passes: prices first, so the charge rule can be stated against the
        # day's own cheapest price rather than against a hard-coded period list.
        for s in range(self.n_slots):
            h = s // self.slots_per_hour
            per = dpm[:, h].astype(int)
            if dhp is None:
                self.slot_prices[:, s] = dpr[np.arange(n_days), per - 1]
            else:
                self.slot_prices[:, s] = dhp[np.arange(n_days), h]
        self._day_min_price = self.slot_prices.min(axis=1)
        for s in range(self.n_slots):
            # Charge where the day's price is at its own minimum, not where the
            # period carries one of a fixed set of numbers. Keying the rule to
            # period numbers -- np.isin(per, (4, 5, 6)) -- silently disabled the
            # battery in the mild pattern, whose three periods P4, P5 and P6 are
            # all in that set: it charged around the clock and never discharged.
            # The consequence was previously read as storage being unprofitable
            # at small spreads and attributed to round-trip losses, of which this
            # model has none. It was an artefact of the rule.
            self.charge_bess_flag[:, s] = np.isclose(
                self.slot_prices[:, s], self._day_min_price, rtol=0.0, atol=1e-12)
        self.priced_hourly = dhp is not None

        if self.has_bess:
            self._load_bess_curves(bess_capacity_kwh)
        self._cache = {}

    # --- reused verbatim from v1 ---
    def _load_bess_curves(self, capacity_kwh):
        base_path = Path(__file__).parent.parent / 'Files'
        cf = base_path / 'PowerForChargeTime.csv'; df = base_path / 'PowerForDischargeTime.csv'
        if cf.exists():
            data = np.loadtxt(str(cf), delimiter=','); self.bess_charge_soc = data[:, 0]
            self.bess_charge_energy = data[:, 1] * (capacity_kwh / 90.0)
        else:
            self.bess_charge_soc = np.linspace(0, 1, 20); peak = capacity_kwh * 0.19
            self.bess_charge_energy = np.where(self.bess_charge_soc < 0.8, peak,
                                               peak * (1.0 - self.bess_charge_soc) / 0.2)
        if df.exists():
            data = np.loadtxt(str(df), delimiter=','); self.bess_discharge_soc = data[:, 0]
            self.bess_discharge_energy = data[:, 1] * (capacity_kwh / 90.0)
        else:
            self.bess_discharge_soc = np.linspace(0, 1, 20)
            self.bess_discharge_energy = np.linspace(0, capacity_kwh * 0.34, 20)

    def _precompute_vehicles(self, mc_data, max_vehicles):
        total_max_veh = sum(max_vehicles); n_days = len(mc_data)
        self.n_days = n_days; self.total_max_veh = total_max_veh
        self.veh_energy_to_charge = np.zeros((n_days, total_max_veh))
        self.veh_energy_per_slot = np.zeros((n_days, total_max_veh))
        self.veh_slots_needed = np.zeros((n_days, total_max_veh), dtype=int)
        self.veh_count = np.zeros(n_days, dtype=int)
        self.veh_total_demand = np.zeros(n_days)
        # A vehicle asking for more than its pack holds gets what fits. That
        # truncation is physical, but it is also the only place demand can go
        # missing, so it is measured and exposed rather than left silent.
        truncated_veh = 0; total_veh = 0; lost_kwh = 0.0; wanted_kwh = 0.0
        for d, day_data in enumerate(mc_data):
            n_veh = len(day_data); self.veh_count[d] = n_veh
            for v, (vid, mp, cap_kwh, soc_i, soc_t, e_to_charge) in enumerate(
                    build_day_vehicles(day_data)):
                total_veh += 1
                asked_kwh = day_data[vid] / WH_PER_KWH
                wanted_kwh += asked_kwh
                if asked_kwh > e_to_charge + 1e-12:
                    truncated_veh += 1; lost_kwh += asked_kwh - e_to_charge
                # A vehicle cannot draw more than the charger delivers, so the
                # window it needs is sized on the rate it will ACTUALLY get.
                #
                # Sizing it on the vehicle's own rating reserved a window for a
                # speed the station never reaches: a 150 kW car on a 70 kW
                # charger got half the time it needs, and the energy that did
                # not fit inside that window was simply lost. That, and not the
                # socket count, is what capped coverage around 0.84 while the
                # daily energy was abundant -- a fleet of six fast vehicles
                # asking 242 kWh topped out lower than one of nine mixed
                # vehicles asking 350 kWh.
                effective_kw = min(mp, self.charger_power_kw)
                if effective_kw > 0 and e_to_charge > 0:
                    ct_min = round((e_to_charge / effective_kw) * 60.0, 2)
                    slots = int(np.ceil(ct_min / self.slot_duration_min))
                else:
                    slots = 0
                self.veh_energy_to_charge[d, v] = e_to_charge
                self.veh_energy_per_slot[d, v] = e_to_charge / max(slots, 1)
                self.veh_slots_needed[d, v] = slots
            self.veh_total_demand[d] = self.veh_energy_to_charge[d, :n_veh].sum()
        self.battery_truncation_vehicle_fraction = truncated_veh / total_veh if total_veh else 0.0
        self.battery_truncation_energy_fraction = lost_kwh / wanted_kwh if wanted_kwh else 0.0
        # int64 contiguous copies for the numba occupancy kernel
        self._vc64 = np.ascontiguousarray(self.veh_count.astype(np.int64))
        self._vsn64 = np.ascontiguousarray(self.veh_slots_needed.astype(np.int64))

    def _build_occupancy_one(self, arrival_slots, occ, base):
        """Fill occ[base+d] for all days for a single schedule (matches v1).
        numba-JIT'd greedy fill; Python fallback if numba is unavailable."""
        arr = np.ascontiguousarray(np.asarray(arrival_slots, dtype=np.int64))
        if _HAS_NUMBA:
            _build_occ_numba(arr, self._vc64, self._vsn64,
                             np.int64(self.socket_limit), np.int64(self.n_slots), occ, np.int64(base))
            return
        for d in range(self.n_days):
            slot_count = np.zeros(self.n_slots, dtype=int); n_veh = self.veh_count[d]
            order = sorted(range(n_veh), key=lambda v: arr[v])
            for v in order:
                start = min(int(arr[v]), self.n_slots - 1)
                needed = self.veh_slots_needed[d, v]
                for s in range(start, min(start + needed, self.n_slots)):
                    if slot_count[s] < self.socket_limit:
                        occ[base + d, v, s] = True; slot_count[s] += 1
                    else:
                        break

    def _simulate(self, occ, prices, charge_flags, veh_eps, veh_total_demand,
                  return_raw=False, soc_open=None, return_soc=False):
        """Vectorized 96-slot sim over leading axis D (= n_days or pop*n_days).
        occ:(D,V,S) bool; prices,charge_flags:(D,S); veh_eps:(D,V); veh_total_demand:(D,)"""
        D = occ.shape[0]
        cep = self.charger_energy_per_slot
        if not self.has_bess:
            bess_soc = None
            soc_open_arr = None
        else:
            if soc_open is None:
                soc_open = self.soc_open if self.soc_open is not None else self.bess_initial_soc
            soc_open_arr = np.broadcast_to(np.asarray(soc_open, dtype=float), (D,)).copy()
            bess_soc = soc_open_arr.copy()
        # Forecast of the energy the day will serve outside the cheap window.
        # It is the mean over realisations, which the operator can form from the
        # committed schedule and the demand distribution; the per-realisation
        # figure would be foresight the battery cannot have.
        if self.has_bess:
            dear = ~charge_flags                               # (D, S)
            per_slot = np.einsum("dvs,dv->ds", occ.astype(np.float64),
                                 np.asarray(veh_eps, dtype=np.float64))
            charge_budget = float((per_slot * dear).sum(axis=1).mean())
        else:
            charge_budget = 0.0
        if _HAS_NUMBA:
            _dummy = np.zeros(1)
            (total_cost, total_served, peak_grid, self._dbg_perslot,
             soc_close, refill, purchased) = _slot_loop_numba(
                    np.ascontiguousarray(occ),
                    np.ascontiguousarray(prices, dtype=np.float64),
                    np.ascontiguousarray(charge_flags),
                    np.ascontiguousarray(veh_eps, dtype=np.float64),
                    soc_open_arr if self.has_bess else np.zeros(D),
                    float(cep),
                    float(self.grid_budget) if self.grid_budget is not None else 0.0,
                    self.grid_budget is not None,
                    bool(self.has_bess),
                    float(self.bess_capacity) if self.has_bess else 1.0,
                    float(self.slots_per_hour),
                    float(self.hours_per_slot),
                    self.bess_charge_soc if self.has_bess else _dummy,
                    self.bess_charge_energy if self.has_bess else _dummy,
                    float(charge_budget))
            if self.has_bess:
                bess_soc = soc_close
            self._last_peak_grid_powers = peak_grid.tolist()
            # kWh actually bought from the grid, which is what makes the
            # tariff price-band check possible: cost per SERVED kWh can exceed
            # the dearest tariff period once storage absorbs energy, but cost
            # per PURCHASED kWh cannot.
            self._last_energy_purchased = purchased.tolist()
            return self._finish(total_cost, total_served, bess_soc, soc_open_arr,
                                prices, charge_flags, veh_total_demand,
                                return_raw, return_soc, refill=refill)

        taken = np.zeros(D)
        total_cost = np.zeros(D); total_served = np.zeros(D)
        self._dbg_perslot = np.zeros((D, self.n_slots))
        peak_grid = np.zeros(D)
        purchased = np.zeros(D)
        for s in range(self.n_slots):
            price = prices[:, s]                                   # (D,)
            # ratio below is scalar per day, so energy served is simply
            # ratio * total_demand: no (D, V) temporary is needed anywhere.
            total_demand = np.einsum('dv,dv->d', veh_eps, occ[:, :, s])

            if not self.has_bess:
                available = np.full(D, cep)
                if self.grid_budget is not None:
                    available = np.minimum(available, self.grid_budget)
                ratio = np.where(total_demand > available, available / np.maximum(total_demand, 1e-10), 1.0)
                energy_served = ratio * total_demand
                total_cost += energy_served * price
                total_served += energy_served
                self._dbg_perslot[:, s] = energy_served
                purchased += energy_served
                peak_grid = np.maximum(peak_grid, energy_served / self.hours_per_slot)
                continue

            # One path for every slot. The uniform-slot branch that used to
            # sit here existed only for bit-parity with the v1 evaluator, which
            # no longer exists; the masked path below reduces to it exactly when
            # a slot's charge flag is uniform, and one path is what makes the
            # loop compilable.
            cflag = charge_flags[:, s]
            not_full = bess_soc < 1.0
            bess_want = np.interp(bess_soc, self.bess_charge_soc,
                                  self.bess_charge_energy) * not_full
            # Never buy more than fits, nor more than the day is forecast to
            # need outside the cheap window (see the compiled path).
            bess_want = np.minimum(bess_want, (1.0 - bess_soc) * self.bess_capacity)
            bess_want = np.minimum(bess_want, np.maximum(0.0, charge_budget - taken))
            if self.grid_budget is not None:
                veh_need = np.minimum(total_demand, cep)
                bess_charge = np.minimum(bess_want,
                                         np.maximum(0.0, self.grid_budget - veh_need))
            else:
                bess_charge = bess_want
            bess_charge = np.where(cflag, bess_charge, 0.0)
            bess_soc = np.minimum(bess_soc + bess_charge / self.bess_capacity, 1.0)
            taken = taken + bess_charge
            total_cost += bess_charge * price
            avail_charge = np.full(D, cep)
            if self.grid_budget is not None:
                avail_charge = np.minimum(avail_charge,
                                          np.maximum(0.0, self.grid_budget - bess_charge))
            bess_avail = bess_soc * self.bess_capacity / self.slots_per_hour
            avail_disch = np.full(D, cep)
            if self.grid_budget is not None:
                avail_disch = np.minimum(avail_disch, self.grid_budget)
            avail_disch = avail_disch + bess_avail
            available = np.where(cflag, avail_charge, avail_disch)
            ratio = np.where(total_demand > available,
                             available / np.maximum(total_demand, 1e-10), 1.0)
            energy_served = ratio * total_demand
            bd = np.where(~cflag, np.minimum(bess_avail, energy_served), 0.0)
            bess_soc = np.maximum(bess_soc - bd / self.bess_capacity, 0.0)
            total_cost += (energy_served - bd) * price
            grid_slot = energy_served - bd + bess_charge
            total_served += energy_served
            self._dbg_perslot[:, s] = energy_served
            purchased += grid_slot
            peak_grid = np.maximum(peak_grid, grid_slot / self.hours_per_slot)

        self._last_peak_grid_powers = peak_grid.tolist()
        self._last_energy_purchased = purchased.tolist()
        return self._finish(total_cost, total_served, bess_soc, soc_open_arr,
                            prices, charge_flags, veh_total_demand,
                            return_raw, return_soc)

    def _finish(self, total_cost, total_served, bess_soc, soc_open_arr,
                prices, charge_flags, veh_total_demand, return_raw, return_soc,
                refill=None):
        """Settle the storage balance, then turn totals into metrics.

        Shared by the compiled and the numpy slot loops so the two cannot drift
        apart on the part that decides what a day costs.
        """
        # "stationary" without a solved law has no law to close the balance
        # in distribution, so it must settle like "cyclic" rather than fall
        # back to the leaking behaviour. Only an explicit "open" skips this.
        needs_settling = (self.boundary == "cyclic"
                          or (self.boundary == "stationary" and self.soc_open is None))
        if self.has_bess and needs_settling:
            # Settle the day's net change in stored energy at the price the
            # battery would have refilled at (the cheapest charge-flagged slot
            # of that day). Without this the reported cost per kWh leaves the
            # tariff's purchasable band and varies with bess_initial_soc.
            if refill is None:
                # numpy fallback: the compiled loop returns this already, and
                # the (D, S) temporary here is why it is worth returning.
                refill = np.where(charge_flags, prices, np.inf).min(axis=1)
                no_charge_slot = ~np.isfinite(refill)
                if no_charge_slot.any():
                    refill = np.where(no_charge_slot, prices.min(axis=1), refill)
            net_drawn = (soc_open_arr - bess_soc) * self.bess_capacity
            total_cost += net_drawn * refill

        er = np.where(veh_total_demand > 0, total_served / veh_total_demand, 1.0)
        ac = np.where(total_served > 0, total_cost / total_served, 0.0)
        if return_soc:
            return er, ac, bess_soc
        if return_raw:
            return er, ac, total_cost, total_served
        return er, ac

    def _occupancy(self, design_variables):
        occ = np.zeros((self.n_days, self.total_max_veh, self.n_slots), dtype=bool)
        self._build_occupancy_one(
            np.array([int(v) for v in design_variables[:self.total_max_veh]],
                     dtype=int), occ, 0)
        return occ

    @staticmethod
    def _ks_distance(a, b):
        """Two-sample Kolmogorov-Smirnov statistic, without a scipy dependency."""
        a = np.sort(np.asarray(a, dtype=float))
        b = np.sort(np.asarray(b, dtype=float))
        grid = np.concatenate([a, b])
        fa = np.searchsorted(a, grid, side="right") / a.size
        fb = np.searchsorted(b, grid, side="right") / b.size
        return float(np.max(np.abs(fa - fb)))

    def solve_stationary_soc(self, design_variables, n_iter=40, tol=5e-3,
                             mean_tol=1e-9, seed=0):
        """Find the opening-SOC law that reproduces itself under this schedule.

        The opening state of charge is a random variable, but not a free one:
        it is the previous day's closing state. Drawing it from a law that does
        not respond to the schedule would let the optimiser empty the battery
        for nothing, which is the boundary leak in statistical clothing. We
        therefore require a fixed point, law(SOC_close) = law(SOC_open), which
        is the stationary law of the daily chain obtained without simulating
        the chain.

        The update permutes the closing SOCs across days rather than resampling
        them, so the empirical law is carried over exactly and only the pairing
        with a day is randomised; that keeps days exchangeable and avoids adding
        resampling noise to the convergence test.

        Convergence is judged on two criteria, not one. The KS distance bounds
        the sup-distance between the opening and closing CDFs; it says nothing
        whatever about their means, so a law can satisfy it while drifting in
        its first moment. That drift is not a diagnostic curiosity: because no
        settlement is applied under this design, `mean(soc_close) -
        mean(soc_open)` is energy that was served and never bought, and it
        reaches the reported cost per kWh directly. Accepting a law on KS
        alone published four cost-per-served-kWh values below the tariff's
        cheapest hour, at drifts fifty to four hundred times inside this
        function's own KS tolerance.

        Where the daily map forgets its opening state -- the regime the
        campaign runs in, with no binding grid limit -- the law is an exact
        fixed point after one application, KS reaches zero and the drift is
        zero to machine precision, so `mean_tol` costs nothing. Where it does
        not forget, the iteration plateaus and no amount of further iteration
        closes either criterion: the drift is then a sampling fluctuation of
        the random day-to-SOC pairing, zero-mean across pairings but not for
        any one of them. That case is reported rather than silently accepted.

        Returns (soc_open, history) where history holds the KS distance at
        each iteration. Three further results are left on the instance:

          stationary_ks          the final KS distance
          stationary_residual    the final mean drift, in SOC units; multiply
                                 by the capacity for the unpriced energy per day
          stationary_converged   whether BOTH criteria were met

        A caller that prices the result must check `stationary_converged`. It
        is False rather than an exception because a campaign that has run for
        hours should record a bad cell, not lose the other cells with it.
        """
        self.stationary_ks = 0.0
        self.stationary_residual = 0.0
        self.stationary_converged = True
        if not self.has_bess:
            return None, []
        rng = np.random.default_rng(seed)
        occ = self._occupancy(design_variables)
        soc_open = np.full(self.n_days, float(np.mean(self.bess_initial_soc)))
        history = []
        ks = drift = float("inf")
        for _ in range(n_iter):
            _, _, soc_close = self._simulate(
                occ, self.slot_prices, self.charge_bess_flag,
                self.veh_energy_per_slot, self.veh_total_demand,
                soc_open=soc_open, return_soc=True)
            ks = self._ks_distance(soc_open, soc_close)
            # Signed, and measured before the permutation: this is the net
            # energy the pool draws out of storage and never pays for.
            drift = float(np.mean(soc_close) - np.mean(soc_open))
            history.append(ks)
            soc_open = rng.permutation(soc_close)
            if ks < tol and abs(drift) <= mean_tol:
                break
        self.stationary_ks = float(ks)
        self.stationary_residual = float(drift)
        self.stationary_converged = bool(ks < tol and abs(drift) <= mean_tol)
        self.soc_open = soc_open
        self._cache.clear()
        return soc_open, history

    def simulate_chain(self, design_variables, soc_init=None, order=None):
        """Walk the days in calendar order, carrying the state of charge.

        This is the reference design: it assumes nothing about the day boundary
        because there is no boundary to assume about. Stored energy is carried
        from one day to the next exactly as it would be on a real site, so the
        only unclosed balance in the whole horizon is at its two ends, worth
        1/n_days of a battery.

        It cannot be vectorised over days, which is the price: the day loop is
        sequential. It is therefore meant for validating the cheap designs, not
        for use inside the optimiser.

        Returns (er, ac, soc_close) with one entry per day, in the given order.
        """
        if not self.has_bess:
            er, ac = self._simulate(
                self._occupancy(design_variables), self.slot_prices,
                self.charge_bess_flag, self.veh_energy_per_slot,
                self.veh_total_demand)
            return er, ac, None

        occ = self._occupancy(design_variables)
        idx = np.arange(self.n_days) if order is None else np.asarray(order)
        soc = float(self.bess_initial_soc if soc_init is None else soc_init)
        er = np.empty(self.n_days)
        ac = np.empty(self.n_days)
        soc_close = np.empty(self.n_days)

        settle, self.boundary = self.boundary, "open"   # no settlement: the
        try:                                            # chain closes itself
            for pos, d in enumerate(idx):
                sl = slice(d, d + 1)
                e, c, sc = self._simulate(
                    occ[sl], self.slot_prices[sl], self.charge_bess_flag[sl],
                    self.veh_energy_per_slot[sl], self.veh_total_demand[sl],
                    soc_open=soc, return_soc=True)
                er[pos], ac[pos], soc_close[pos] = e[0], c[0], sc[0]
                soc = float(sc[0])
        finally:
            self.boundary = settle
        return er, ac, soc_close

    def _stats(self, er, ac):
        if _HAS_NUMBA:
            a, b, c, d, e, f = _stats_numba(np.ascontiguousarray(er),
                                            np.ascontiguousarray(ac))
            return float(a), float(b), float(c), float(d), float(e), float(f)
        return (float(np.percentile(er, 5)), float(np.percentile(ac, 95)),
                float(np.mean(er)), float(np.std(er)), float(np.mean(ac)), float(np.std(ac)))

    def evaluate(self, design_variables):
        key = tuple(int(v) for v in design_variables[:self.total_max_veh])
        if key in self._cache:
            return self._cache[key]
        occ = np.zeros((self.n_days, self.total_max_veh, self.n_slots), dtype=bool)
        self._build_occupancy_one(np.array(key, dtype=int), occ, 0)
        er, ac = self._simulate(occ, self.slot_prices, self.charge_bess_flag,
                                self.veh_energy_per_slot, self.veh_total_demand)
        res = self._stats(er, ac)
        self._cache[key] = res
        return res

    def evaluate_batch(self, X):
        """X: list/array of P schedules. Returns list of P stat-tuples."""
        P = len(X); nd = self.n_days
        occ = np.zeros((P * nd, self.total_max_veh, self.n_slots), dtype=bool)
        for p in range(P):
            self._build_occupancy_one(np.array([int(v) for v in X[p][:self.total_max_veh]], dtype=int),
                                      occ, p * nd)
        # Tile per-day arrays across the population
        prices = np.tile(self.slot_prices, (P, 1))
        cflags = np.tile(self.charge_bess_flag, (P, 1)) if self.has_bess else np.zeros((P*nd, self.n_slots), bool)
        veh_eps = np.tile(self.veh_energy_per_slot, (P, 1))
        vtd = np.tile(self.veh_total_demand, P)
        soc0 = (np.tile(self.soc_open, P) if self.soc_open is not None else None)
        er, ac = self._simulate(occ, prices, cflags, veh_eps, vtd, soc_open=soc0)
        er = er.reshape(P, nd); ac = ac.reshape(P, nd)
        return [self._stats(er[p], ac[p]) for p in range(P)]


def create_fast_evaluator(mc_data, max_vehicles, ems, boundary="cyclic"):
    """Factory: build the evaluator from an existing EMS object.

    Replaces the v1 factory of the same name. The default is "cyclic", which
    is safe for a search: it never creates or destroys stored energy and costs
    nothing. Reported results should be re-evaluated under "stationary", the
    design validated against the chronological reference. Pass "open" only to
    reproduce previously published numbers.
    """
    charger = ems.getCharger()
    battery = charger.getBattery()

    period_map = np.array([ems.getPeriod(h) for h in range(24)], dtype=int)
    prices = np.array([ems.getPrice(p) for p in range(1, 7)], dtype=float)

    return FastMCEvaluatorV2(
        mc_data=mc_data,
        max_vehicles=max_vehicles,
        charger_power_kw=charger.getMaxPower(),
        socket_limit=charger.getSocketsNumber(),
        bess_capacity_kwh=battery.getBatteryCapacity() if battery else 0.0,
        bess_initial_soc=(ems._EMS__initial_charger_soc_battery
                          if battery else 0.0),
        day_period_map=period_map,
        day_prices=prices,
        grid_contract_power=ems.getGridContractPower(),
        boundary=boundary,
    )
