"""Run the full dispatch experiment and write every reported number to disk.

Pipeline:
  1. Load 2023 (training) and 2024 (evaluation) day-ahead prices.
  2. Fit the price forecasters on 2023 only.
  3. Score forecast accuracy on 2024.
  4. Dispatch the battery over 2024 under every policy.
  5. Compare each against the perfect-foresight upper bound.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --quick    # two months, for a fast check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from battery.data import load_years  # noqa: E402
from battery.forecast import (  # noqa: E402
    GradientBoostingForecaster,
    SeasonalNaiveForecaster,
    evaluate_forecast,
)
from battery.model import BatterySpec, check_feasibility  # noqa: E402
from battery.optimize import perfect_foresight_dispatch  # noqa: E402
from battery.policies import (  # noqa: E402
    FixedSchedulePolicy,
    ThresholdPolicy,
    apply_schedule,
    hours_of_day,
)
from battery.simulate import capture_rate, rolling_horizon_dispatch  # noqa: E402
from battery.utils import load_config, save_json, set_seed, setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="Use only the first 60 days")
    p.add_argument("--config", default="configs/default.yaml")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config(args.config)
    set_seed(config["seed"])

    spec = BatterySpec.from_config(config)
    opt_cfg = config["optimization"]
    solver = opt_cfg["solver"]

    log.info(
        "Battery: %.0f MWh / %.0f MW | usable %.0f MWh | %.1f h duration | eta_rt %.2f",
        spec.energy_capacity_mwh, spec.power_mw, spec.usable_capacity_mwh,
        spec.duration_hours, spec.round_trip_efficiency,
    )

    # --- 1. data ----------------------------------------------------------
    years = load_years(config)
    train = years[config["data"]["train_year"]]
    test = years[config["data"]["test_year"]]

    if args.quick:
        limit = 60 * 24
        test = type(test)(
            timestamps=test.timestamps[:limit],
            prices_eur_mwh=test.prices_eur_mwh[:limit],
            bidding_zone=test.bidding_zone,
            license_info=test.license_info,
        )

    log.info("Train %d: %s", config["data"]["train_year"], train.stats())
    log.info("Test  %d: %s", config["data"]["test_year"], test.stats())

    actuals = test.values
    test_hours = hours_of_day(test.timestamps)

    # --- 2 & 3. forecasting ----------------------------------------------
    # Fitted on the training year alone. The evaluation year is never seen.
    forecasts: dict[str, np.ndarray] = {}
    forecast_metrics = []

    naive = SeasonalNaiveForecaster()
    forecasts[naive.name] = naive.predict(actuals, test.timestamps)
    forecast_metrics.append(
        evaluate_forecast(forecasts[naive.name], actuals, naive.name).metrics()
    )

    gbm_cfg = config["forecast"]["gbm"]
    gbm = GradientBoostingForecaster(
        lags=config["forecast"]["lags"],
        rolling_windows=config["forecast"]["rolling_windows"],
        n_estimators=gbm_cfg["n_estimators"],
        learning_rate=gbm_cfg["learning_rate"],
        max_depth=gbm_cfg["max_depth"],
        seed=config["seed"],
    )
    log.info("Fitting gradient boosting on %d training hours ...", len(train))
    gbm.fit(train.values, train.timestamps)
    forecasts[gbm.name] = gbm.predict(actuals, test.timestamps)
    forecast_metrics.append(
        evaluate_forecast(forecasts[gbm.name], actuals, gbm.name).metrics()
    )

    for m in forecast_metrics:
        log.info("Forecast %-22s MAE %6.2f  RMSE %6.2f", m["name"], m["mae_eur_mwh"],
                 m["rmse_eur_mwh"])

    # --- 4. dispatch ------------------------------------------------------
    results: dict[str, dict] = {}

    log.info("Solving perfect-foresight upper bound ...")
    oracle = perfect_foresight_dispatch(actuals, spec, solver_name=solver)
    oracle_revenue = oracle.net_revenue_eur(spec)
    violations = check_feasibility(oracle, spec)
    if violations:
        log.error("Oracle dispatch violates physics: %s", violations[:5])
        return 1
    results["perfect_foresight"] = oracle.summary(spec) | {"capture_rate": 1.0}
    log.info("Perfect foresight: EUR %.0f", oracle_revenue)

    log.info("Fixed schedule baseline ...")
    fixed = FixedSchedulePolicy()
    fixed_result = apply_schedule(
        fixed.plan(actuals, test_hours, spec), spec, actuals
    )
    results["fixed_schedule"] = fixed_result.summary(spec) | {
        "capture_rate": round(capture_rate(fixed_result.net_revenue_eur(spec), oracle_revenue), 4)
    }

    log.info("Threshold policy: fitting quantiles on the training year ...")
    threshold = ThresholdPolicy.fit(train.values, spec)
    log.info("  chosen quantiles: %.2f / %.2f", threshold.low_quantile, threshold.high_quantile)
    threshold_result = apply_schedule(
        threshold.plan(actuals, test_hours, spec), spec, actuals
    )
    results["threshold"] = threshold_result.summary(spec) | {
        "capture_rate": round(
            capture_rate(threshold_result.net_revenue_eur(spec), oracle_revenue), 4
        ),
        "low_quantile": threshold.low_quantile,
        "high_quantile": threshold.high_quantile,
    }

    for name, prediction in forecasts.items():
        log.info("Rolling-horizon MILP driven by %s ...", name)
        run = rolling_horizon_dispatch(
            prediction,
            actuals,
            spec,
            horizon_hours=opt_cfg["horizon_hours"],
            commit_hours=opt_cfg["commit_hours"],
            solver_name=solver,
            time_limit_seconds=opt_cfg["time_limit_seconds"],
        )
        violations = check_feasibility(run, spec)
        if violations:
            log.error("%s dispatch violates physics: %s", name, violations[:5])
            return 1
        results[f"milp_{name}"] = run.summary(spec) | {
            "capture_rate": round(capture_rate(run.net_revenue_eur(spec), oracle_revenue), 4)
        }

    # --- 5. report --------------------------------------------------------
    save_json(
        {
            "battery": {
                "energy_capacity_mwh": spec.energy_capacity_mwh,
                "power_mw": spec.power_mw,
                "usable_capacity_mwh": spec.usable_capacity_mwh,
                "round_trip_efficiency": spec.round_trip_efficiency,
                "degradation_cost_eur_per_mwh": spec.degradation_cost_eur_per_mwh,
            },
            "data": {
                "train": train.stats(),
                "test": test.stats(),
                "license": test.license_info,
            },
            "forecast_metrics": forecast_metrics,
            "dispatch": results,
        },
        "reports/metrics_dispatch.json",
    )

    np.save(
        Path(__file__).resolve().parents[1] / "reports" / "forecast_errors.npy",
        forecasts[gbm.name] - actuals,
    )

    print()
    print("| Policy | Net revenue (EUR) | Capture rate | Cycles/day |")
    print("|---|---:|---:|---:|")
    order = [
        "fixed_schedule", "threshold",
        "milp_seasonal_naive_168h", "milp_gradient_boosting",
        "perfect_foresight",
    ]
    for key in order:
        if key not in results:
            continue
        r = results[key]
        print(f"| {key} | {r['net_revenue_eur']:,.0f} | {r['capture_rate']:.1%} | "
              f"{r['cycles_per_day']:.2f} |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
