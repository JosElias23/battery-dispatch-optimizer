"""Tests for the battery physics and the economics built on it.

Every test here guards a failure mode that would produce a plausible revenue
figure rather than an error.
"""

import math

import pytest

from battery.model import BatterySpec, DispatchResult, check_feasibility


@pytest.fixture
def spec():
    return BatterySpec(
        energy_capacity_mwh=100.0,
        power_mw=25.0,
        round_trip_efficiency=0.86,
        soc_min_fraction=0.1,
        soc_max_fraction=0.9,
        initial_soc_fraction=0.5,
        degradation_cost_eur_per_mwh=3.0,
    )


class TestBatterySpec:
    def test_efficiencies_multiply_to_the_round_trip(self, spec):
        assert spec.eta_charge * spec.eta_discharge == pytest.approx(
            spec.round_trip_efficiency
        )

    def test_usable_capacity_respects_depth_of_discharge(self, spec):
        assert spec.usable_capacity_mwh == pytest.approx(80.0)

    def test_duration_is_usable_energy_over_power(self, spec):
        assert spec.duration_hours == pytest.approx(80.0 / 25.0)

    def test_breakeven_accounts_for_losses_and_wear(self, spec):
        # Buying at 100 EUR/MWh: the round trip loses 14%, so 1 MWh delivered
        # cost 100/0.86 = 116.28 to buy, plus 3.00 of degradation.
        assert spec.breakeven_spread_eur_per_mwh(100.0) == pytest.approx(
            100.0 / 0.86 + 3.0
        )

    def test_breakeven_exceeds_the_buy_price(self, spec):
        """A lossy asset can never break even on a zero spread."""
        for price in (10.0, 50.0, 200.0):
            assert spec.breakeven_spread_eur_per_mwh(price) > price

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"energy_capacity_mwh": 0.0},
            {"power_mw": -1.0},
            {"round_trip_efficiency": 1.5},
            {"round_trip_efficiency": 0.0},
            {"soc_min_fraction": 0.9, "soc_max_fraction": 0.1},
            {"initial_soc_fraction": 0.99},
            {"degradation_cost_eur_per_mwh": -1.0},
        ],
    )
    def test_invalid_specifications_are_rejected(self, kwargs):
        base = {
            "energy_capacity_mwh": 100.0,
            "power_mw": 25.0,
            "round_trip_efficiency": 0.86,
            "soc_min_fraction": 0.1,
            "soc_max_fraction": 0.9,
            "initial_soc_fraction": 0.5,
        }
        with pytest.raises(ValueError):
            BatterySpec(**{**base, **kwargs})


class TestDispatchEconomics:
    def test_round_trip_at_a_flat_price_loses_money(self, spec):
        """The single most important economic sanity check.

        Buying and selling the same energy at one price must lose exactly the
        round-trip loss plus wear. A model that breaks even here has its
        efficiency applied on only one leg, which silently overstates revenue
        by several percent on every result in the project.
        """
        price = 100.0
        # Charge 25 MW for one hour, then discharge what that put in.
        stored = spec.eta_charge * 25.0
        discharge = stored * spec.eta_discharge

        result = DispatchResult(
            charge_mw=[25.0, 0.0],
            discharge_mw=[0.0, discharge],
            soc_mwh=[
                spec.initial_soc_mwh + stored,
                spec.initial_soc_mwh + stored - discharge / spec.eta_discharge,
            ],
            prices_eur_mwh=[price, price],
        )

        expected_gross = price * (discharge - 25.0)
        assert result.gross_revenue_eur == pytest.approx(expected_gross)
        assert result.gross_revenue_eur < 0
        assert result.net_revenue_eur(spec) < result.gross_revenue_eur

    def test_energy_returned_matches_round_trip_efficiency(self, spec):
        drawn = 25.0
        returned = spec.eta_charge * drawn * spec.eta_discharge
        assert returned / drawn == pytest.approx(spec.round_trip_efficiency)

    def test_degradation_is_charged_on_discharge_only(self, spec):
        result = DispatchResult(
            charge_mw=[10.0, 0.0],
            discharge_mw=[0.0, 8.0],
            soc_mwh=[0.0, 0.0],
            prices_eur_mwh=[50.0, 50.0],
        )
        assert result.degradation_cost_eur(spec) == pytest.approx(3.0 * 8.0)

    def test_equivalent_cycles_uses_usable_not_nameplate_capacity(self, spec):
        result = DispatchResult(
            charge_mw=[0.0],
            discharge_mw=[80.0],
            soc_mwh=[0.0],
            prices_eur_mwh=[50.0],
        )
        assert result.equivalent_full_cycles(spec) == pytest.approx(1.0)


class TestFeasibilityChecker:
    """The checker is the last line of defence; it must actually catch things."""

    def test_a_valid_schedule_passes(self, spec):
        result = DispatchResult(
            charge_mw=[10.0, 0.0],
            discharge_mw=[0.0, 5.0],
            soc_mwh=[],
            prices_eur_mwh=[50.0, 80.0],
        )
        soc = spec.initial_soc_mwh
        trace = []
        for c, d in zip(result.charge_mw, result.discharge_mw, strict=True):
            soc += spec.eta_charge * c - d / spec.eta_discharge
            trace.append(soc)
        result.soc_mwh = trace
        assert check_feasibility(result, spec) == []

    def test_exceeding_rated_power_is_caught(self, spec):
        result = DispatchResult(
            charge_mw=[spec.power_mw + 5.0],
            discharge_mw=[0.0],
            soc_mwh=[],
            prices_eur_mwh=[50.0],
        )
        result.soc_mwh = [spec.initial_soc_mwh + spec.eta_charge * result.charge_mw[0]]
        assert any("exceeds rated power" in v for v in check_feasibility(result, spec))

    def test_simultaneous_charge_and_discharge_is_caught(self, spec):
        result = DispatchResult(
            charge_mw=[10.0],
            discharge_mw=[10.0],
            soc_mwh=[],
            prices_eur_mwh=[-50.0],
        )
        soc = spec.initial_soc_mwh + spec.eta_charge * 10.0 - 10.0 / spec.eta_discharge
        result.soc_mwh = [soc]
        assert any("simultaneous" in v for v in check_feasibility(result, spec))

    def test_overcharging_past_the_soc_limit_is_caught(self, spec):
        # Twelve hours of charging at full power cannot fit in the usable band.
        n = 12
        result = DispatchResult(
            charge_mw=[spec.power_mw] * n,
            discharge_mw=[0.0] * n,
            soc_mwh=[],
            prices_eur_mwh=[10.0] * n,
        )
        soc = spec.initial_soc_mwh
        trace = []
        for c in result.charge_mw:
            soc += spec.eta_charge * c
            trace.append(soc)
        result.soc_mwh = trace
        assert any("above maximum" in v for v in check_feasibility(result, spec))

    def test_inconsistent_reported_soc_is_caught(self, spec):
        """A schedule whose SoC trace disagrees with its own power profile."""
        result = DispatchResult(
            charge_mw=[10.0],
            discharge_mw=[0.0],
            soc_mwh=[spec.initial_soc_mwh],  # claims nothing happened
            prices_eur_mwh=[50.0],
        )
        assert any("disagrees with" in v for v in check_feasibility(result, spec))

    def test_negative_power_is_caught(self, spec):
        result = DispatchResult(
            charge_mw=[-5.0],
            discharge_mw=[0.0],
            soc_mwh=[spec.initial_soc_mwh - spec.eta_charge * 5.0],
            prices_eur_mwh=[50.0],
        )
        assert any("negative charge" in v for v in check_feasibility(result, spec))


def test_efficiency_of_one_is_lossless():
    """Boundary case: a perfect battery returns exactly what it took."""
    spec = BatterySpec(energy_capacity_mwh=10.0, power_mw=5.0, round_trip_efficiency=1.0)
    assert spec.eta_charge == 1.0
    assert math.isclose(spec.breakeven_spread_eur_per_mwh(50.0), 50.0)
