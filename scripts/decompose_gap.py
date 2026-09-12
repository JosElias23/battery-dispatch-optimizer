"""Split the gap to perfect foresight into forecast error and horizon cost.

Finding 4 of the write-up says the 14.3% shortfall against perfect foresight is
"entirely attributable to forecast error, every euro of it", and that "no
improvement to the solver, the formulation or the horizon can recover any of
it". The experiment as built does not establish that, because the two arms
being subtracted differ in **two** ways rather than one:

    perfect foresight   one optimisation over all 8,784 hours, true prices
    MILP + forecast     rolling 48-hour horizon, 24 hours committed, forecast

So the difference mixes *not knowing the prices* with *not being allowed to plan
past the day after tomorrow*. A full-year optimiser can carry energy across a
quiet week into a volatile one; a 48-hour window cannot see that far, and the
state of charge it hands to the next window is chosen without knowing what comes
after. That is a cost of the horizon, and it is present even with perfect
prices.

This script adds the arm that separates them: the same rolling-horizon dispatch,
the same window and commitment, driven by the **actual** prices.

    A  full-year perfect foresight          upper bound
    B  rolling horizon, true prices         A - B is the cost of the horizon
    C  rolling horizon, GB forecast         B - C is the cost of forecast error

Nothing here is new machinery -- arm B is `rolling_horizon_dispatch` called with
`actuals` in the forecast slot, which is exactly what the Monte Carlo does with
a zero error path. It simply had never been run.

**The result vindicates the claim it was written to challenge.** The horizon
costs nothing: arm B earns EUR 2,574,513 against the bound's EUR 2,573,659, so
the entire EUR 367,799 shortfall is forecast error and none of it is the
48-hour window. In hindsight the reason is in the asset: a 100 MWh / 25 MW
battery has a four-hour duration, so it fills and empties inside a day and never
needs to carry energy across a week. A horizon twice the cycle length is already
long enough, and the claim in finding 4 is now measured rather than argued.

One thing did fall out of it. Arm B **beats** the bound, by EUR 855. An upper
bound that a real policy exceeds is not an upper bound, and the reason is in
`perfect_foresight_dispatch`: it solves the year in fortnight chunks, each
starting from the previous chunk's final state of charge. Its own docstring
says that "can only ever understate the optimum" -- and also, two paragraphs
earlier, that "no real policy can beat this". Both cannot hold, and the second
is the one that is wrong: a rolling 48-hour window sees across the fortnight
seams that the chunked bound cannot.

The discrepancy is 0.03% and moves no conclusion. What it changes is a word:
capture rates in this repository are fractions of a chunked perfect-foresight
benchmark, not of the most an operator could possibly have earned.

Usage:
    python scripts/decompose_gap.py

Writes reports/metrics_gap_decomposition.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from battery.data import load_years  # noqa: E402
from battery.forecast import GradientBoostingForecaster  # noqa: E402
from battery.model import BatterySpec  # noqa: E402
from battery.optimize import perfect_foresight_dispatch  # noqa: E402
from battery.simulate import rolling_horizon_dispatch  # noqa: E402
from battery.utils import load_config, save_json, setup_logging  # noqa: E402


def main() -> int:
    log = setup_logging()
    config = load_config()
    spec = BatterySpec.from_config(config)
    opt = config["optimization"]
    solver = opt.get("solver", "HiGHS")

    years = load_years(config)
    train = years[config["data"]["train_year"]]
    test = years[config["data"]["test_year"]]
    actuals = test.values

    gbm_cfg = config["forecast"]["gbm"]
    gbm = GradientBoostingForecaster(
        lags=config["forecast"]["lags"],
        rolling_windows=config["forecast"]["rolling_windows"],
        n_estimators=gbm_cfg["n_estimators"],
        learning_rate=gbm_cfg["learning_rate"],
        max_depth=gbm_cfg["max_depth"],
        seed=config["seed"],
    )
    gbm.fit(train.values, train.timestamps)
    forecast = gbm.predict(actuals, test.timestamps)

    def rolling(prices):
        return rolling_horizon_dispatch(
            prices, actuals, spec,
            horizon_hours=opt["horizon_hours"],
            commit_hours=opt["commit_hours"],
            solver_name=solver,
            time_limit_seconds=opt["time_limit_seconds"],
        ).net_revenue_eur(spec)

    log.info("A: full-year perfect foresight ...")
    a = perfect_foresight_dispatch(actuals, spec, solver_name=solver).net_revenue_eur(spec)

    log.info("B: rolling horizon on true prices ...")
    b = rolling(actuals)

    log.info("C: rolling horizon on the gradient-boosted forecast ...")
    c = rolling(forecast)

    total = a - c
    horizon_cost = a - b
    forecast_cost = b - c

    report = {
        "horizon_hours": opt["horizon_hours"],
        "commit_hours": opt["commit_hours"],
        "full_year_perfect_foresight_eur": round(a, 2),
        "rolling_horizon_true_prices_eur": round(b, 2),
        "rolling_horizon_forecast_eur": round(c, 2),
        "total_gap_eur": round(total, 2),
        "attributable_to_horizon_eur": round(horizon_cost, 2),
        "attributable_to_forecast_error_eur": round(forecast_cost, 2),
        "horizon_share_of_gap": round(horizon_cost / total, 4) if total else None,
        "forecast_share_of_gap": round(forecast_cost / total, 4) if total else None,
        "note": ("Arm B is the same rolling-horizon dispatch as the deployed "
                 "policy, handed the realised prices. Any shortfall it shows "
                 "against the full-year bound is the price of planning 48 hours "
                 "ahead and committing 24, and no forecaster can recover it."),
    }
    save_json(report, "reports/metrics_gap_decomposition.json")

    print()
    print("| Arm | Net revenue | Gap to bound |")
    print("|---|---:|---:|")
    print(f"| A  full-year perfect foresight | EUR {a:,.0f} | - |")
    print(f"| B  rolling horizon, true prices | EUR {b:,.0f} | EUR {a - b:,.0f} |")
    print(f"| C  rolling horizon, GB forecast | EUR {c:,.0f} | EUR {a - c:,.0f} |")
    print()
    print(f"Of the EUR {total:,.0f} gap: EUR {horizon_cost:,.0f} "
          f"({100 * horizon_cost / total:.1f}%) is the rolling horizon and "
          f"EUR {forecast_cost:,.0f} ({100 * forecast_cost / total:.1f}%) is "
          f"forecast error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
