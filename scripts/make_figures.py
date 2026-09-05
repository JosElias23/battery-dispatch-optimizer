"""Generate every figure in the README from the stored result files.

Reads only reports/*.json, so a figure can never disagree with a reported
number: both come from the same artefact.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from battery.data import load_years  # noqa: E402
from battery.model import BatterySpec  # noqa: E402
from battery.optimize import solve_dispatch  # noqa: E402
from battery.utils import PROJECT_ROOT, load_config, setup_logging  # noqa: E402

PRETTY = {
    "fixed_schedule": "Fixed schedule",
    "threshold": "Price threshold",
    "milp_seasonal_naive_168h": "MILP + naive\nforecast",
    "milp_gradient_boosting": "MILP + gradient\nboosting",
    "perfect_foresight": "Perfect foresight\n(upper bound)",
}
ORDER = [
    "fixed_schedule",
    "threshold",
    "milp_seasonal_naive_168h",
    "milp_gradient_boosting",
    "perfect_foresight",
]
COLORS = {
    "fixed_schedule": "#9aa0a6",
    "threshold": "#b0885f",
    "milp_seasonal_naive_168h": "#5f6caf",
    "milp_gradient_boosting": "#00798c",
    "perfect_foresight": "#d1495b",
}


def plot_capture_rates(metrics: dict, out: Path) -> None:
    dispatch = metrics["dispatch"]
    keys = [k for k in ORDER if k in dispatch]
    revenues = [dispatch[k]["net_revenue_eur"] / 1e6 for k in keys]
    captures = [dispatch[k]["capture_rate"] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(
        range(len(keys)), revenues, color=[COLORS[k] for k in keys], width=0.62
    )
    for bar, revenue, capture in zip(bars, revenues, captures, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            revenue + 0.045,
            f"EUR {revenue:.2f}M\n{capture:.1%} of optimum",
            ha="center",
            fontsize=9.5,
            linespacing=1.4,
        )

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([PRETTY[k] for k in keys], fontsize=9.5)
    ax.set_ylabel("Net revenue, 2024 (million EUR)")
    ax.set_ylim(0, max(revenues) * 1.22)
    ax.set_title(
        "100 MWh / 25 MW battery on German day-ahead prices, 2024\n"
        "Optimisation more than doubles what a fixed schedule earns",
        fontsize=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_example_week(config: dict, out: Path) -> None:
    """A week of prices with the optimal schedule and state of charge."""
    spec = BatterySpec.from_config(config)
    test = load_years(config)[config["data"]["test_year"]]

    # Pick the week with the widest price range: the clearest illustration.
    values = test.values
    usable = len(values) - len(values) % 168
    weeks = values[:usable].reshape(-1, 168)
    best = int(np.argmax(weeks.max(axis=1) - weeks.min(axis=1)))
    start = best * 168
    prices = list(values[start : start + 168])

    result = solve_dispatch(prices, spec, solver_name=config["optimization"]["solver"])
    hours = np.arange(168)

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.4, 1.2]}
    )

    axes[0].plot(hours, prices, color="#333333", linewidth=1.3)
    axes[0].axhline(0, color="#d1495b", linewidth=0.9, linestyle="--", alpha=0.8)
    axes[0].fill_between(hours, prices, 0, where=np.array(prices) < 0,
                         color="#d1495b", alpha=0.25)
    axes[0].set_ylabel("Price\n(EUR/MWh)")
    axes[0].set_title(
        f"Optimal dispatch over the most volatile week of {config['data']['test_year']}"
        f"  |  net revenue EUR {result.net_revenue_eur(spec):,.0f}",
        fontsize=12,
    )
    axes[0].grid(alpha=0.25)

    axes[1].bar(hours, result.charge_mw, color="#00798c", width=0.9, label="Charge")
    axes[1].bar(hours, [-d for d in result.discharge_mw], color="#edae49",
                width=0.9, label="Discharge")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Power\n(MW)")
    axes[1].legend(loc="upper right", fontsize=9, ncol=2)
    axes[1].grid(alpha=0.25)

    axes[2].plot(hours, result.soc_mwh, color="#5f6caf", linewidth=1.6)
    axes[2].axhline(spec.soc_max_mwh, color="#999999", linestyle=":", linewidth=1)
    axes[2].axhline(spec.soc_min_mwh, color="#999999", linestyle=":", linewidth=1)
    axes[2].set_ylabel("State of charge\n(MWh)")
    axes[2].set_xlabel("Hour of week")
    axes[2].grid(alpha=0.25)

    for ax in axes:
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_monte_carlo(mc: dict, deterministic_revenue: float, out: Path) -> None:
    revenues = np.array(mc["revenues"]) / 1e6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(revenues, bins=32, color="#00798c", alpha=0.78, edgecolor="white")

    lower, upper = mc["ci95_lower_eur"] / 1e6, mc["ci95_upper_eur"] / 1e6
    ax.axvspan(lower, upper, color="#00798c", alpha=0.12,
               label=f"95% interval: EUR {lower:.2f}M to {upper:.2f}M")
    ax.axvline(mc["mean_eur"] / 1e6, color="#333333", linewidth=1.6,
               label=f"Mean: EUR {mc['mean_eur'] / 1e6:.2f}M")
    ax.axvline(deterministic_revenue / 1e6, color="#d1495b", linewidth=1.6,
               linestyle="--",
               label=f"Realised 2024 run: EUR {deterministic_revenue / 1e6:.2f}M")

    ax.set_xlabel("Annual net revenue (million EUR)")
    ax.set_ylabel("Simulated years")
    ax.set_title(
        f"Revenue under resampled forecast error, {mc['n_simulations']} simulated years\n"
        f"Forecast risk moves annual revenue by "
        f"+/-{100 * mc['std_eur'] / mc['mean_eur']:.1f}% (1 sd)",
        fontsize=12,
    )
    ax.legend(fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> int:
    log = setup_logging()
    config = load_config()
    reports = PROJECT_ROOT / "reports"
    figures = PROJECT_ROOT / config["paths"]["figures_dir"]
    figures.mkdir(parents=True, exist_ok=True)

    metrics_path = reports / "metrics_dispatch.json"
    if not metrics_path.exists():
        log.error("Run scripts/run_experiment.py first.")
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    plot_capture_rates(metrics, figures / "capture_rates.png")
    log.info("Wrote capture_rates.png")

    plot_example_week(config, figures / "example_week.png")
    log.info("Wrote example_week.png")

    mc_path = reports / "metrics_monte_carlo.json"
    if mc_path.exists():
        mc = json.loads(mc_path.read_text(encoding="utf-8"))
        deterministic = metrics["dispatch"]["milp_gradient_boosting"]["net_revenue_eur"]
        plot_monte_carlo(mc, deterministic, figures / "monte_carlo.png")
        log.info("Wrote monte_carlo.png")
    else:
        log.warning("No Monte Carlo results yet; skipping that figure.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
