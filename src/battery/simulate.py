"""Rolling-horizon operation and Monte Carlo risk analysis.

Rolling horizon
---------------
A real operator does not solve the year in one shot. Each day it plans over the
next `horizon_hours` using whatever forecast it has, commits only the first
`commit_hours`, then re-plans tomorrow with a day more information.

Committing less than the full horizon is what stops the plan collapsing at the
window edge. Energy still in the battery at the end of a planning window is
worth nothing *inside* that window, so an optimiser that commits everything it
plans will empty the battery every night regardless of tomorrow's prices. With
a 48-hour horizon and a 24-hour commitment, the discarded second day absorbs
that end effect.

The crucial separation: **plans are made on forecast prices, revenue is booked
at realised prices.** Scoring a forecast-driven plan against the forecast that
produced it measures nothing but the optimiser's arithmetic.

Monte Carlo
-----------
A single revenue figure hides how much of it was luck. `monte_carlo_revenue`
resamples the observed forecast errors to build alternative price paths the
year could plausibly have taken, and reports the resulting distribution.

Errors are resampled in day-long blocks. Hourly forecast errors are strongly
autocorrelated -- a model that misses a price spike misses the whole evening,
not one isolated hour -- and resampling hour by hour would break that
dependence, cancel the errors out, and produce a confidence interval far
narrower than reality.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from battery.model import BatterySpec, DispatchResult
from battery.optimize import solve_dispatch


def rolling_horizon_dispatch(
    forecast_prices: Sequence[float],
    actual_prices: Sequence[float],
    spec: BatterySpec,
    horizon_hours: int = 48,
    commit_hours: int = 24,
    dt_hours: float = 1.0,
    solver_name: str = "HiGHS",
    time_limit_seconds: int = 60,
) -> DispatchResult:
    """Operate the battery day by day on forecasts, and value it at real prices.

    Returns a DispatchResult whose schedule came from `forecast_prices` but
    whose `prices_eur_mwh` are `actual_prices`, so every revenue property on it
    reports money that could actually have been earned.
    """
    if len(forecast_prices) != len(actual_prices):
        raise ValueError(
            f"{len(forecast_prices)} forecast hours vs {len(actual_prices)} actual hours"
        )
    if commit_hours > horizon_hours:
        raise ValueError("commit_hours cannot exceed horizon_hours")

    n = len(actual_prices)
    charge: list[float] = []
    discharge: list[float] = []
    soc_trace: list[float] = []
    soc = spec.initial_soc_mwh
    total_solve_time = 0.0

    for start in range(0, n, commit_hours):
        window = list(forecast_prices[start : start + horizon_hours])
        if not window:
            break

        result = solve_dispatch(
            window,
            spec,
            initial_soc_mwh=soc,
            dt_hours=dt_hours,
            solver_name=solver_name,
            time_limit_seconds=time_limit_seconds,
            enforce_complementarity=True,
        )
        total_solve_time += result.solve_seconds

        # Commit only the first `commit_hours`, and never past the end of data.
        take = min(commit_hours, n - start, len(window))
        charge.extend(result.charge_mw[:take])
        discharge.extend(result.discharge_mw[:take])
        soc_trace.extend(result.soc_mwh[:take])
        soc = result.soc_mwh[take - 1]

    return DispatchResult(
        charge_mw=charge,
        discharge_mw=discharge,
        soc_mwh=soc_trace,
        prices_eur_mwh=list(actual_prices),  # valued at reality, not at the forecast
        dt_hours=dt_hours,
        status="RollingHorizon",
        solve_seconds=total_solve_time,
    )


def capture_rate(policy_revenue: float, oracle_revenue: float) -> float:
    """Share of the perfect-foresight optimum a policy achieved.

    The headline metric of this project. Absolute revenue depends on how
    volatile the year happened to be and is not comparable across markets or
    years; the fraction of the attainable optimum is.
    """
    if oracle_revenue <= 0:
        raise ValueError("Oracle revenue must be positive to define a capture rate")
    return policy_revenue / oracle_revenue


def block_bootstrap_errors(
    errors: np.ndarray,
    n_hours: int,
    block_hours: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample forecast errors in contiguous blocks, preserving autocorrelation."""
    if len(errors) < block_hours:
        raise ValueError(f"Need at least {block_hours} error observations")

    n_blocks = int(np.ceil(n_hours / block_hours))
    starts = rng.integers(0, len(errors) - block_hours + 1, size=n_blocks)
    sampled = np.concatenate([errors[s : s + block_hours] for s in starts])
    return sampled[:n_hours]


def monte_carlo_revenue(
    actual_prices: Sequence[float],
    forecast_errors: np.ndarray,
    spec: BatterySpec,
    n_simulations: int = 500,
    block_hours: int = 24,
    horizon_hours: int = 48,
    commit_hours: int = 24,
    seed: int = 42,
    solver_name: str = "HiGHS",
    progress_every: int = 25,
) -> dict:
    """Distribution of annual revenue under resampled forecast error.

    Each simulation builds a synthetic forecast by adding a block-bootstrapped
    error path to the realised prices, runs the full rolling-horizon dispatch
    against it, and books revenue at the realised prices. The spread of the
    result answers the question a risk committee actually asks: not "what did
    this earn" but "what range could it have earned".
    """
    rng = np.random.default_rng(seed)
    actuals = np.asarray(actual_prices, dtype=float)
    revenues: list[float] = []

    for i in range(n_simulations):
        noise = block_bootstrap_errors(forecast_errors, len(actuals), block_hours, rng)
        synthetic_forecast = actuals + noise

        result = rolling_horizon_dispatch(
            synthetic_forecast,
            actuals,
            spec,
            horizon_hours=horizon_hours,
            commit_hours=commit_hours,
            solver_name=solver_name,
        )
        revenues.append(result.net_revenue_eur(spec))

        if progress_every and (i + 1) % progress_every == 0:
            print(f"[monte carlo] {i + 1}/{n_simulations} simulations")

    values = np.asarray(revenues)
    return {
        "n_simulations": n_simulations,
        "block_hours": block_hours,
        "mean_eur": round(float(values.mean()), 2),
        "std_eur": round(float(values.std(ddof=1)), 2),
        "min_eur": round(float(values.min()), 2),
        "max_eur": round(float(values.max()), 2),
        "p05_eur": round(float(np.quantile(values, 0.05)), 2),
        "p50_eur": round(float(np.quantile(values, 0.50)), 2),
        "p95_eur": round(float(np.quantile(values, 0.95)), 2),
        "ci95_lower_eur": round(float(np.quantile(values, 0.025)), 2),
        "ci95_upper_eur": round(float(np.quantile(values, 0.975)), 2),
        "revenues": [round(v, 2) for v in revenues],
    }
