"""Physical and economic model of a grid-connected battery.

Everything the optimiser is allowed to know about the asset lives here, so the
physics can be tested independently of the optimisation.

Sign and efficiency conventions, fixed once and used everywhere:

    charge[t]     power drawn FROM the grid, MW, >= 0
    discharge[t]  power delivered TO the grid, MW, >= 0

    energy into the cells   = eta_charge * charge[t] * dt
    energy out of the cells = discharge[t] * dt / eta_discharge

Delivering 1 MWh to the grid therefore costs 1 / (eta_c * eta_d) = 1 / eta_rt
MWh drawn from it, which is the definition of round-trip efficiency. Getting
this convention wrong in either direction produces a model that looks fine and
silently over- or under-states revenue by several percent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BatterySpec:
    """Physical limits and economics of the storage asset."""

    energy_capacity_mwh: float
    power_mw: float
    round_trip_efficiency: float
    soc_min_fraction: float = 0.0
    soc_max_fraction: float = 1.0
    initial_soc_fraction: float = 0.5
    degradation_cost_eur_per_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.energy_capacity_mwh <= 0:
            raise ValueError("energy_capacity_mwh must be positive")
        if self.power_mw <= 0:
            raise ValueError("power_mw must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if not 0 <= self.soc_min_fraction < self.soc_max_fraction <= 1:
            raise ValueError("Require 0 <= soc_min_fraction < soc_max_fraction <= 1")
        if not self.soc_min_fraction <= self.initial_soc_fraction <= self.soc_max_fraction:
            raise ValueError("initial_soc_fraction must lie within the SoC bounds")
        if self.degradation_cost_eur_per_mwh < 0:
            raise ValueError("degradation_cost_eur_per_mwh must be non-negative")

    # --- derived quantities ------------------------------------------------

    @property
    def eta_charge(self) -> float:
        """Charging efficiency, taken as the square root of the round trip.

        Splitting the loss symmetrically is the standard convention when only
        the round-trip figure is published, as is almost always the case.
        """
        return math.sqrt(self.round_trip_efficiency)

    @property
    def eta_discharge(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_fraction * self.energy_capacity_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_fraction * self.energy_capacity_mwh

    @property
    def usable_capacity_mwh(self) -> float:
        """Energy actually available for trading, after depth-of-discharge limits."""
        return self.soc_max_mwh - self.soc_min_mwh

    @property
    def initial_soc_mwh(self) -> float:
        return self.initial_soc_fraction * self.energy_capacity_mwh

    @property
    def duration_hours(self) -> float:
        """Hours to discharge the usable capacity at rated power."""
        return self.usable_capacity_mwh / self.power_mw

    def breakeven_spread_eur_per_mwh(self, buy_price: float) -> float:
        """Minimum sell price that makes one round trip worth doing.

        Buying at `buy_price` and selling one MWh back to the grid requires
        1 / eta_rt MWh of purchased energy, and incurs the degradation charge.
        Any spread below this destroys value, which is precisely why a battery
        does not simply arbitrage every wiggle in the price curve.
        """
        return buy_price / self.round_trip_efficiency + self.degradation_cost_eur_per_mwh

    @classmethod
    def from_config(cls, config: dict) -> BatterySpec:
        return cls(**config["battery"])


@dataclass
class DispatchResult:
    """The outcome of operating the battery over a price series."""

    charge_mw: list[float]
    discharge_mw: list[float]
    soc_mwh: list[float]
    prices_eur_mwh: list[float]
    dt_hours: float = 1.0
    status: str = "unknown"
    solve_seconds: float = 0.0

    @property
    def gross_revenue_eur(self) -> float:
        """Energy sold minus energy bought, valued at the realised price."""
        return sum(
            price * (d - c) * self.dt_hours
            for price, c, d in zip(
                self.prices_eur_mwh, self.charge_mw, self.discharge_mw, strict=True
            )
        )

    @property
    def energy_discharged_mwh(self) -> float:
        return sum(self.discharge_mw) * self.dt_hours

    @property
    def energy_charged_mwh(self) -> float:
        return sum(self.charge_mw) * self.dt_hours

    def degradation_cost_eur(self, spec: BatterySpec) -> float:
        return spec.degradation_cost_eur_per_mwh * self.energy_discharged_mwh

    def net_revenue_eur(self, spec: BatterySpec) -> float:
        """What the operator actually keeps. This is the reported metric."""
        return self.gross_revenue_eur - self.degradation_cost_eur(spec)

    def equivalent_full_cycles(self, spec: BatterySpec) -> float:
        """Throughput expressed in full charge-discharge cycles.

        The standard unit for battery duty, and the one warranties are written
        in. A number far above roughly one cycle per day signals a model that
        is ignoring degradation.
        """
        return self.energy_discharged_mwh / spec.usable_capacity_mwh

    def summary(self, spec: BatterySpec) -> dict:
        n_hours = len(self.prices_eur_mwh) * self.dt_hours
        return {
            "net_revenue_eur": round(self.net_revenue_eur(spec), 2),
            "gross_revenue_eur": round(self.gross_revenue_eur, 2),
            "degradation_cost_eur": round(self.degradation_cost_eur(spec), 2),
            "energy_charged_mwh": round(self.energy_charged_mwh, 2),
            "energy_discharged_mwh": round(self.energy_discharged_mwh, 2),
            "equivalent_full_cycles": round(self.equivalent_full_cycles(spec), 2),
            "cycles_per_day": round(self.equivalent_full_cycles(spec) / (n_hours / 24), 3),
            "hours": int(n_hours),
            "status": self.status,
            "solve_seconds": round(self.solve_seconds, 2),
        }


def check_feasibility(
    result: DispatchResult, spec: BatterySpec, tolerance: float = 1e-6
) -> list[str]:
    """Re-verify a dispatch against the physics, independently of the solver.

    A solver reporting "Optimal" only means it satisfied the constraints it was
    given. If those constraints were written incorrectly, the answer is
    optimal for the wrong problem. This function re-derives the state of charge
    from scratch and checks every physical limit, so a modelling error surfaces
    as a violation instead of as revenue.
    """
    violations: list[str] = []
    dt = result.dt_hours
    soc = spec.initial_soc_mwh

    for t, (c, d) in enumerate(zip(result.charge_mw, result.discharge_mw, strict=True)):
        if c < -tolerance:
            violations.append(f"t={t}: negative charge {c:.6f}")
        if d < -tolerance:
            violations.append(f"t={t}: negative discharge {d:.6f}")
        if c > spec.power_mw + tolerance:
            violations.append(f"t={t}: charge {c:.4f} exceeds rated power {spec.power_mw}")
        if d > spec.power_mw + tolerance:
            violations.append(f"t={t}: discharge {d:.4f} exceeds rated power {spec.power_mw}")
        if c > tolerance and d > tolerance:
            violations.append(f"t={t}: simultaneous charge {c:.4f} and discharge {d:.4f}")

        soc += spec.eta_charge * c * dt - d * dt / spec.eta_discharge

        if soc < spec.soc_min_mwh - 1e-4:
            violations.append(f"t={t}: SoC {soc:.4f} below minimum {spec.soc_min_mwh}")
        if soc > spec.soc_max_mwh + 1e-4:
            violations.append(f"t={t}: SoC {soc:.4f} above maximum {spec.soc_max_mwh}")

    if result.soc_mwh:
        drift = abs(soc - result.soc_mwh[-1])
        if drift > 1e-3:
            violations.append(
                f"Reported final SoC {result.soc_mwh[-1]:.4f} disagrees with "
                f"recomputed {soc:.4f} (drift {drift:.6f})"
            )

    return violations
