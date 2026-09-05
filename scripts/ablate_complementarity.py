"""Quantify the revenue invented by dropping the complementarity constraint.

The MILP forbids charging and discharging in the same hour via a binary. That
constraint is routinely described as redundant -- doing both at once is never
optimal, so the linear relaxation should be exact and much faster to solve.

The argument assumes prices are positive. This script tests it on a market
where 5.2% of hours cleared below zero.

Usage:
    python scripts/ablate_complementarity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from battery.data import load_years  # noqa: E402
from battery.model import BatterySpec, check_feasibility  # noqa: E402
from battery.optimize import perfect_foresight_dispatch, solve_dispatch  # noqa: E402
from battery.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402

CHUNK_HOURS = 24 * 14


def relaxed_dispatch(prices, spec, solver_name):
    """Perfect-foresight dispatch with the binary removed."""
    from battery.model import DispatchResult

    charge, discharge, soc_trace = [], [], []
    soc = spec.initial_soc_mwh
    elapsed = 0.0
    for start in range(0, len(prices), CHUNK_HOURS):
        window = prices[start : start + CHUNK_HOURS]
        result = solve_dispatch(
            window, spec, initial_soc_mwh=soc,
            solver_name=solver_name, enforce_complementarity=False,
        )
        charge.extend(result.charge_mw)
        discharge.extend(result.discharge_mw)
        soc_trace.extend(result.soc_mwh)
        soc = result.soc_mwh[-1]
        elapsed += result.solve_seconds

    return DispatchResult(
        charge, discharge, soc_trace, list(prices),
        status="Relaxed", solve_seconds=elapsed,
    )


def main() -> int:
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    spec = BatterySpec.from_config(config)
    solver = config["optimization"]["solver"]

    test = load_years(config)[config["data"]["test_year"]]
    prices = test.values

    log.info("Solving with the complementarity constraint ...")
    constrained = perfect_foresight_dispatch(prices, spec, solver_name=solver)

    log.info("Solving without it ...")
    relaxed = relaxed_dispatch(prices, spec, solver)

    honest = constrained.net_revenue_eur(spec)
    inflated = relaxed.net_revenue_eur(spec)

    # Hours in which the relaxed solution does both at once. These are exactly
    # the hours the physical asset could not have operated.
    overlap = [
        (t, c, d, float(prices[t]))
        for t, (c, d) in enumerate(zip(relaxed.charge_mw, relaxed.discharge_mw, strict=True))
        if c > 1e-6 and d > 1e-6
    ]
    negative_hours = [row for row in overlap if row[3] < 0]

    violations = check_feasibility(relaxed, spec)

    log.info("With constraint:    EUR %12.2f", honest)
    log.info("Without constraint: EUR %12.2f", inflated)
    log.info("Fictitious revenue: EUR %12.2f (%.2f%%)",
             inflated - honest, 100 * (inflated - honest) / honest)
    log.info("Hours running both directions at once: %d (of which %d at negative prices)",
             len(overlap), len(negative_hours))
    log.info("Physical violations detected in the relaxed schedule: %d", len(violations))

    result = {
        "constrained_net_revenue_eur": round(honest, 2),
        "relaxed_net_revenue_eur": round(inflated, 2),
        "fictitious_revenue_eur": round(inflated - honest, 2),
        "fictitious_revenue_pct": round(100 * (inflated - honest) / honest, 3),
        "simultaneous_hours": len(overlap),
        "simultaneous_hours_at_negative_prices": len(negative_hours),
        "negative_price_hours_in_year": int((prices < 0).sum()),
        "physical_violations_in_relaxed_schedule": len(violations),
        "constrained_solve_seconds": round(constrained.solve_seconds, 2),
        "relaxed_solve_seconds": round(relaxed.solve_seconds, 2),
        "example_hours": [
            {
                "hour_index": t,
                "charge_mw": round(c, 3),
                "discharge_mw": round(d, 3),
                "price_eur_mwh": round(p, 2),
            }
            for t, c, d, p in sorted(overlap, key=lambda r: r[3])[:10]
        ],
    }
    save_json(result, "reports/metrics_complementarity_ablation.json")

    if overlap:
        worst = min(overlap, key=lambda r: r[3])
        log.info(
            "Most extreme hour: price %.2f EUR/MWh, charging %.2f MW while "
            "discharging %.2f MW simultaneously.",
            worst[3], worst[1], worst[2],
        )
    else:
        log.info("The relaxation never ran both directions at once on this data.")

    print()
    print(f"Fictitious revenue from the relaxation: EUR {inflated - honest:,.0f} "
          f"({100 * (inflated - honest) / honest:.2f}% of honest revenue)")
    print(f"Impossible hours: {len(overlap)} / {len(prices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
