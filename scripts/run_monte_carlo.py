"""Monte Carlo analysis of revenue risk under forecast uncertainty.

The headline experiment reports one revenue figure for one realised year. That
number conflates skill with luck. This script asks the question a risk
committee would: given a forecaster of this quality, what range of annual
revenue should be expected?

Forecast errors observed in 2024 are resampled in day-long blocks to build
alternative price paths the year could plausibly have taken, the full
rolling-horizon dispatch is run against each, and revenue is booked at the
realised prices throughout.

Usage:
    python scripts/run_monte_carlo.py
    python scripts/run_monte_carlo.py --n-simulations 100 --workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from battery.data import load_years  # noqa: E402
from battery.model import BatterySpec  # noqa: E402
from battery.simulate import monte_carlo_revenue  # noqa: E402
from battery.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-simulations", type=int, default=None)
    p.add_argument("--workers", type=int, default=None, help="Processes (default: all cores)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    spec = BatterySpec.from_config(config)
    mc_cfg = config["monte_carlo"]
    opt_cfg = config["optimization"]
    n_sims = args.n_simulations or mc_cfg["n_simulations"]

    errors_path = PROJECT_ROOT / "reports" / "forecast_errors.npy"
    if not errors_path.exists():
        log.error("Run scripts/run_experiment.py first to produce forecast errors.")
        return 1
    errors = np.load(errors_path)

    test = load_years(config)[config["data"]["test_year"]]
    actuals = test.values

    log.info(
        "Forecast error distribution: mean %.2f, sd %.2f, MAE %.2f EUR/MWh",
        errors.mean(), errors.std(), np.abs(errors).mean(),
    )
    log.info("Running %d simulated years on %s workers ...",
             n_sims, args.workers or "all")

    result = monte_carlo_revenue(
        actuals,
        errors,
        spec,
        n_simulations=n_sims,
        block_hours=mc_cfg["block_hours"],
        horizon_hours=opt_cfg["horizon_hours"],
        commit_hours=opt_cfg["commit_hours"],
        seed=config["seed"],
        solver_name=opt_cfg["solver"],
        n_workers=args.workers,
    )

    result["forecast_error_mae_eur_mwh"] = round(float(np.abs(errors).mean()), 3)
    result["forecast_error_std_eur_mwh"] = round(float(errors.std()), 3)
    save_json(result, "reports/metrics_monte_carlo.json")

    print()
    print(f"Simulations           : {result['n_simulations']}")
    print(f"Mean revenue          : EUR {result['mean_eur']:,.0f}")
    print(f"Standard deviation    : EUR {result['std_eur']:,.0f}")
    print(f"95% interval          : EUR {result['ci95_lower_eur']:,.0f} "
          f"to {result['ci95_upper_eur']:,.0f}")
    print(f"5th / 95th percentile : EUR {result['p05_eur']:,.0f} / {result['p95_eur']:,.0f}")
    print(f"Relative spread       : {100 * result['std_eur'] / result['mean_eur']:.2f}% of mean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
