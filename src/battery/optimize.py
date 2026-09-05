"""Mixed-integer linear program for optimal battery dispatch.

Decision variables, one set per hour t:

    c[t]  charge power drawn from the grid          MW, in [0, P]
    d[t]  discharge power delivered to the grid     MW, in [0, P]
    s[t]  state of charge at the end of hour t      MWh, in [SoC_min, SoC_max]
    y[t]  1 if the unit is charging, 0 otherwise    binary

Objective, maximised:

    sum_t  price[t] * (d[t] - c[t]) * dt  -  degradation_cost * sum_t d[t] * dt

Subject to energy balance, rated power, and the complementarity constraint

    c[t] <= P * y[t]        d[t] <= P * (1 - y[t])

which is the only reason this is an integer program rather than a linear one.


Why the binary is not optional
------------------------------
Textbook formulations frequently drop `y` on the argument that charging and
discharging at the same time is never optimal, so the constraint is redundant
and the relaxation is exact. That argument holds only while prices are
positive. It fails on this dataset, where 5.2% of 2024 hours cleared below
zero.

When price[t] < 0 the operator is *paid* to consume. The relaxed model then
discovers that it can run c[t] and d[t] simultaneously and hold the state of
charge constant while maintaining a net import:

    hold SoC:   eta_c * c = d / eta_d
    net grid:   d - c  =  c * (eta_c * eta_d - 1)  <  0

The round-trip loss becomes a way to absorb paid-for energy indefinitely,
without ever filling the battery. Revenue accrues every hour the price is
negative, and the state of charge never moves. It is a money pump, it is worth
real money in the objective, and it is physically impossible: a battery has one
inverter and cannot import and export at the same instant.

`scripts/ablate_complementarity.py` quantifies exactly how much revenue the
relaxation invents.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import pulp

from battery.model import BatterySpec, DispatchResult


class SolverError(RuntimeError):
    """Raised when the solver fails to return a usable solution."""


def _build_solver(name: str, time_limit: int, msg: bool = False):
    """Return the requested solver, falling back to bundled CBC.

    HiGHS is roughly an order of magnitude faster than CBC on this problem, but
    CBC ships with PuLP and is always present, so the repository stays runnable
    with no extra system dependencies.
    """
    available = set(pulp.listSolvers(onlyAvailable=True))
    if name.upper() in {"HIGHS", "HIGHS_CMD"} and "HiGHS" in available:
        return pulp.HiGHS(msg=msg, timeLimit=time_limit)
    if "PULP_CBC_CMD" in available:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    raise SolverError(f"No usable solver found. Available: {sorted(available)}")


def solve_dispatch(
    prices: Sequence[float],
    spec: BatterySpec,
    initial_soc_mwh: float | None = None,
    dt_hours: float = 1.0,
    solver_name: str = "HiGHS",
    time_limit_seconds: int = 60,
    enforce_complementarity: bool = True,
    terminal_soc_mwh: float | None = None,
) -> DispatchResult:
    """Maximise trading revenue over a price series.

    Parameters
    ----------
    prices:
        Price per hour in EUR/MWh. These are treated as certain. Passing the
        realised prices yields the perfect-foresight upper bound; passing
        forecasts yields an implementable policy.
    initial_soc_mwh:
        Starting energy. Defaults to the spec's initial state of charge.
    terminal_soc_mwh:
        Optional lower bound on the final state of charge. Without it the model
        empties the battery in the last hours of the horizon, because energy
        left in storage is worth nothing inside the window. That is correct for
        the window and wrong for the year, which is why the rolling-horizon
        simulation commits only the first part of each solution.
    enforce_complementarity:
        Keep the binary that forbids simultaneous charge and discharge. Set to
        False only for the ablation; see the module docstring.
    """
    n = len(prices)
    if n == 0:
        raise ValueError("prices must not be empty")

    soc_start = spec.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
    if not (spec.soc_min_mwh - 1e-9 <= soc_start <= spec.soc_max_mwh + 1e-9):
        raise ValueError(
            f"initial_soc_mwh {soc_start} outside "
            f"[{spec.soc_min_mwh}, {spec.soc_max_mwh}]"
        )

    problem = pulp.LpProblem("battery_dispatch", pulp.LpMaximize)

    charge = pulp.LpVariable.dicts("charge", range(n), lowBound=0, upBound=spec.power_mw)
    discharge = pulp.LpVariable.dicts(
        "discharge", range(n), lowBound=0, upBound=spec.power_mw
    )
    soc = pulp.LpVariable.dicts(
        "soc", range(n), lowBound=spec.soc_min_mwh, upBound=spec.soc_max_mwh
    )

    # Objective. Degradation is charged on discharged energy only, so a single
    # round trip is billed once rather than twice.
    problem += pulp.lpSum(
        prices[t] * (discharge[t] - charge[t]) * dt_hours
        - spec.degradation_cost_eur_per_mwh * discharge[t] * dt_hours
        for t in range(n)
    )

    # Energy balance. Losses are applied on the way in and on the way out.
    for t in range(n):
        previous = soc_start if t == 0 else soc[t - 1]
        problem += (
            soc[t]
            == previous
            + spec.eta_charge * charge[t] * dt_hours
            - discharge[t] * dt_hours / spec.eta_discharge
        ), f"energy_balance_{t}"

    if enforce_complementarity:
        mode = pulp.LpVariable.dicts("is_charging", range(n), cat="Binary")
        for t in range(n):
            problem += charge[t] <= spec.power_mw * mode[t], f"charge_mode_{t}"
            problem += discharge[t] <= spec.power_mw * (1 - mode[t]), f"discharge_mode_{t}"

    if terminal_soc_mwh is not None:
        problem += soc[n - 1] >= terminal_soc_mwh, "terminal_soc"

    started = time.perf_counter()
    problem.solve(_build_solver(solver_name, time_limit_seconds))
    elapsed = time.perf_counter() - started

    status = pulp.LpStatus[problem.status]
    if status not in {"Optimal"}:
        raise SolverError(f"Solver returned status {status!r} after {elapsed:.1f}s")

    return DispatchResult(
        charge_mw=[float(charge[t].value() or 0.0) for t in range(n)],
        discharge_mw=[float(discharge[t].value() or 0.0) for t in range(n)],
        soc_mwh=[float(soc[t].value() or 0.0) for t in range(n)],
        prices_eur_mwh=list(prices),
        dt_hours=dt_hours,
        status=status,
        solve_seconds=elapsed,
    )


def perfect_foresight_dispatch(
    prices: Sequence[float],
    spec: BatterySpec,
    dt_hours: float = 1.0,
    solver_name: str = "HiGHS",
    time_limit_seconds: int = 300,
    chunk_hours: int = 24 * 14,
) -> DispatchResult:
    """Upper bound on achievable revenue, given exact knowledge of all prices.

    No real policy can beat this, so it is the natural denominator: reporting
    "captured 78% of the perfect-foresight optimum" says far more than an
    absolute euro figure, which depends entirely on how volatile the year was.

    A full year is one MILP with 8,760 binaries. Rather than solve that
    directly, the year is cut into fortnight-long chunks solved in sequence,
    each starting from the previous chunk's final state of charge. The loss
    from doing so is bounded by what a single battery could carry across a
    chunk boundary, which is at most one full cycle out of roughly 14 -- and it
    keeps the bound honest, because it can only ever *understate* the optimum.
    """
    all_charge: list[float] = []
    all_discharge: list[float] = []
    all_soc: list[float] = []
    total_time = 0.0
    soc_state = spec.initial_soc_mwh

    for start in range(0, len(prices), chunk_hours):
        window = prices[start : start + chunk_hours]
        result = solve_dispatch(
            window,
            spec,
            initial_soc_mwh=soc_state,
            dt_hours=dt_hours,
            solver_name=solver_name,
            time_limit_seconds=time_limit_seconds,
            enforce_complementarity=True,
        )
        all_charge.extend(result.charge_mw)
        all_discharge.extend(result.discharge_mw)
        all_soc.extend(result.soc_mwh)
        total_time += result.solve_seconds
        soc_state = result.soc_mwh[-1]

    return DispatchResult(
        charge_mw=all_charge,
        discharge_mw=all_discharge,
        soc_mwh=all_soc,
        prices_eur_mwh=list(prices),
        dt_hours=dt_hours,
        status="Optimal",
        solve_seconds=total_time,
    )
