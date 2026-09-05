"""Rule-based dispatch policies, and the simulator that keeps them honest.

The MILP needs something to be compared against, or "the optimiser earned
EUR 1.4 M" is a number with no meaning. Two baselines are implemented:

`FixedSchedulePolicy`
    Charge overnight, discharge into the evening peak, same hours every day.
    This is how storage was actually operated before day-ahead optimisation,
    and it still describes a large share of behind-the-meter assets. It ignores
    prices entirely.

`ThresholdPolicy`
    Charge when the price falls below a low quantile of recent history,
    discharge when it rises above a high quantile. Prices matter, but only
    through the present hour: the policy cannot plan.

Both are executed through `apply_schedule`, the same simulator that executes
the optimiser's plan. Every policy is therefore subject to identical physics,
and no policy can win by quietly violating a limit.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from battery.model import BatterySpec, DispatchResult


def apply_schedule(
    desired_power_mw: Sequence[float],
    spec: BatterySpec,
    prices: Sequence[float],
    initial_soc_mwh: float | None = None,
    dt_hours: float = 1.0,
) -> DispatchResult:
    """Execute a desired power profile against the battery's real limits.

    Sign convention: positive means charge (import), negative means discharge
    (export). Requests are clipped, in order, by rated power and by the energy
    actually available or storable in this interval.

    Clipping rather than rejecting is deliberate. A heuristic that asks for
    more than the asset can deliver should get what the asset can deliver, the
    way a real controller behaves -- not an exception, and not a silent free
    pass on the constraint.
    """
    soc = spec.initial_soc_mwh if initial_soc_mwh is None else initial_soc_mwh
    charge_mw: list[float] = []
    discharge_mw: list[float] = []
    soc_trace: list[float] = []

    for request in desired_power_mw:
        charge = discharge = 0.0

        if request > 0:
            charge = min(request, spec.power_mw)
            # Energy that still fits, converted back to grid-side power.
            headroom_mwh = max(0.0, spec.soc_max_mwh - soc)
            max_charge = headroom_mwh / (spec.eta_charge * dt_hours)
            charge = min(charge, max_charge)
        elif request < 0:
            discharge = min(-request, spec.power_mw)
            available_mwh = max(0.0, soc - spec.soc_min_mwh)
            max_discharge = available_mwh * spec.eta_discharge / dt_hours
            discharge = min(discharge, max_discharge)

        soc += spec.eta_charge * charge * dt_hours - discharge * dt_hours / spec.eta_discharge
        # Guard against float drift accumulating past a bound over 8,760 steps.
        soc = min(max(soc, spec.soc_min_mwh), spec.soc_max_mwh)

        charge_mw.append(charge)
        discharge_mw.append(discharge)
        soc_trace.append(soc)

    return DispatchResult(
        charge_mw=charge_mw,
        discharge_mw=discharge_mw,
        soc_mwh=soc_trace,
        prices_eur_mwh=list(prices),
        dt_hours=dt_hours,
        status="Heuristic",
    )


class FixedSchedulePolicy:
    """Charge and discharge at the same clock hours every day, ignoring price.

    Default hours follow the classic load shape: charge in the small hours when
    demand is lowest, discharge into the evening peak. On a modern grid this is
    increasingly wrong -- solar has pushed the cheapest hours to the middle of
    the day -- and quantifying how wrong is part of the point.
    """

    def __init__(
        self,
        charge_hours: Sequence[int] = (0, 1, 2, 3, 4),
        discharge_hours: Sequence[int] = (18, 19, 20, 21),
    ) -> None:
        overlap = set(charge_hours) & set(discharge_hours)
        if overlap:
            raise ValueError(f"Hours cannot both charge and discharge: {sorted(overlap)}")
        self.charge_hours = set(charge_hours)
        self.discharge_hours = set(discharge_hours)

    def plan(self, prices: Sequence[float], hours_of_day: Sequence[int], spec: BatterySpec):
        return [
            spec.power_mw
            if h in self.charge_hours
            else (-spec.power_mw if h in self.discharge_hours else 0.0)
            for h in hours_of_day
        ]


class ThresholdPolicy:
    """Trade against quantiles of a trailing price window.

    Charge below the `low_quantile` of the last `window_hours`, discharge above
    the `high_quantile`. The window is strictly backward-looking, so the policy
    is implementable in real time -- it never reads a price it could not have
    known.

    Quantiles are fitted on the training year by `fit`, never on the evaluation
    year.
    """

    def __init__(
        self,
        low_quantile: float = 0.25,
        high_quantile: float = 0.75,
        window_hours: int = 168,
    ) -> None:
        if not 0 < low_quantile < high_quantile < 1:
            raise ValueError("Require 0 < low_quantile < high_quantile < 1")
        self.low_quantile = low_quantile
        self.high_quantile = high_quantile
        self.window_hours = window_hours

    @classmethod
    def fit(
        cls,
        train_prices: Sequence[float],
        spec: BatterySpec,
        window_hours: int = 168,
        candidates: Sequence[float] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40),
    ) -> ThresholdPolicy:
        """Grid-search the quantile pair that maximises revenue on training data.

        Symmetric quantiles (q, 1-q) are searched rather than the full 2-D grid.
        The asymmetric version fits marginally better on the training year and
        generalises no better, which is exactly the kind of extra parameter a
        baseline should not have.
        """
        best_policy, best_revenue = None, -np.inf
        hours = list(range(len(train_prices)))
        for q in candidates:
            policy = cls(low_quantile=q, high_quantile=1 - q, window_hours=window_hours)
            plan = policy.plan(train_prices, hours, spec)
            revenue = apply_schedule(plan, spec, train_prices).net_revenue_eur(spec)
            if revenue > best_revenue:
                best_policy, best_revenue = policy, revenue
        assert best_policy is not None
        return best_policy

    def plan(self, prices: Sequence[float], hours_of_day: Sequence[int], spec: BatterySpec):
        values = np.asarray(prices, dtype=float)
        actions: list[float] = []

        for t, price in enumerate(values):
            # Strictly backward-looking window; excludes the current hour.
            start = max(0, t - self.window_hours)
            history = values[start:t]
            if len(history) < 24:
                actions.append(0.0)
                continue

            low = float(np.quantile(history, self.low_quantile))
            high = float(np.quantile(history, self.high_quantile))

            if price <= low:
                actions.append(spec.power_mw)
            elif price >= high:
                # Only sell if the price clears the cost of having stored the
                # energy. Without this the policy churns through its cycle life
                # on spreads that do not cover degradation.
                if price >= spec.degradation_cost_eur_per_mwh:
                    actions.append(-spec.power_mw)
                else:
                    actions.append(0.0)
            else:
                actions.append(0.0)

        return actions


def hours_of_day(timestamps, utc_offset_hours: int = 1) -> list[int]:
    """Local hour of day for each timestamp.

    Prices follow human activity, which follows local clocks. Central European
    Time is UTC+1; the summer-time hour is ignored, and the residual is small
    relative to the daily price swing.
    """
    return [(ts.hour + utc_offset_hours) % 24 for ts in timestamps]
