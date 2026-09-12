"""Tests for the MILP and the policy simulator.

These check the optimiser against situations whose answer is known by
inspection. A solver returning "Optimal" proves only that it satisfied the
constraints it was handed; these tests check that those were the right ones.
"""

import numpy as np
import pytest

from battery.model import BatterySpec, check_feasibility
from battery.optimize import perfect_foresight_dispatch, solve_dispatch
from battery.policies import FixedSchedulePolicy, ThresholdPolicy, apply_schedule


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


@pytest.fixture
def lossless():
    """No losses, no wear: makes hand-computed optima exact."""
    return BatterySpec(
        energy_capacity_mwh=100.0,
        power_mw=25.0,
        round_trip_efficiency=1.0,
        soc_min_fraction=0.0,
        soc_max_fraction=1.0,
        initial_soc_fraction=0.5,
        degradation_cost_eur_per_mwh=0.0,
    )


@pytest.fixture
def empty_spec():
    """Starts at the minimum state of charge.

    The default spec begins half full, and that 50 MWh is an endowment the
    optimiser is free to sell without ever having bought it. Over a year that
    is worth about 0.1% of revenue and affects every policy alike, but in a
    two-hour test it dominates the result and hides the property under test.
    """
    return BatterySpec(
        energy_capacity_mwh=100.0,
        power_mw=25.0,
        round_trip_efficiency=0.86,
        soc_min_fraction=0.1,
        soc_max_fraction=0.9,
        initial_soc_fraction=0.1,
        degradation_cost_eur_per_mwh=3.0,
    )


@pytest.fixture
def empty_lossless():
    return BatterySpec(
        energy_capacity_mwh=100.0,
        power_mw=25.0,
        round_trip_efficiency=1.0,
        soc_min_fraction=0.0,
        soc_max_fraction=1.0,
        initial_soc_fraction=0.0,
        degradation_cost_eur_per_mwh=0.0,
    )


class TestOptimalityOnKnownCases:
    def test_flat_prices_produce_no_trading(self, empty_spec):
        """With no spread, every round trip loses money, so the answer is idle."""
        spec = empty_spec
        result = solve_dispatch([80.0] * 24, spec)
        assert result.energy_charged_mwh == pytest.approx(0.0, abs=1e-6)
        assert result.energy_discharged_mwh == pytest.approx(0.0, abs=1e-6)
        assert result.net_revenue_eur(spec) == pytest.approx(0.0, abs=1e-6)

    def test_a_spread_below_breakeven_is_declined(self, empty_spec):
        """Buy at 100, sell at 110: below the 119.28 needed to cover losses."""
        spec = empty_spec
        result = solve_dispatch([100.0] * 4 + [110.0] * 4, spec)
        assert result.energy_charged_mwh == pytest.approx(0.0, abs=1e-6)

    def test_a_spread_above_breakeven_is_taken(self, spec):
        result = solve_dispatch([10.0] * 4 + [300.0] * 4, spec)
        assert result.energy_charged_mwh > 0
        assert result.net_revenue_eur(spec) > 0

    def test_it_buys_low_and_sells_high_not_the_reverse(self, lossless):
        prices = [10.0] * 4 + [200.0] * 4
        result = solve_dispatch(prices, lossless)
        assert sum(result.charge_mw[:4]) > 0
        assert sum(result.charge_mw[4:]) == pytest.approx(0.0, abs=1e-6)
        assert sum(result.discharge_mw[4:]) > 0
        assert sum(result.discharge_mw[:4]) == pytest.approx(0.0, abs=1e-6)

    def test_revenue_matches_the_hand_computed_optimum(self, empty_lossless):
        """Two cheap hours then two expensive ones, every limit known.

        Starting empty and lossless: the battery buys at full power for both
        zero-price hours (2 x 25 MW = 50 MWh), then sells all of it across the
        two hours priced at 100, which is exactly what 25 MW of export
        delivers. Revenue is therefore 50 MWh x 100 EUR/MWh, with nothing paid
        on the way in.
        """
        prices = [0.0, 0.0, 100.0, 100.0]
        result = solve_dispatch(prices, empty_lossless)
        assert result.energy_charged_mwh == pytest.approx(50.0, abs=1e-4)
        assert result.energy_discharged_mwh == pytest.approx(50.0, abs=1e-4)
        assert result.gross_revenue_eur == pytest.approx(5000.0, abs=1e-2)

    def test_more_price_volatility_never_reduces_optimal_revenue(self, spec):
        """A monotonicity property: widening the spread cannot hurt."""
        mild = solve_dispatch([40.0] * 6 + [160.0] * 6, spec).net_revenue_eur(spec)
        wild = solve_dispatch([10.0] * 6 + [400.0] * 6, spec).net_revenue_eur(spec)
        assert wild >= mild


class TestConstraintsHold:
    def test_solution_never_violates_physics(self, spec):
        rng = np.random.default_rng(0)
        prices = list(rng.normal(80, 60, size=72))
        result = solve_dispatch(prices, spec)
        assert check_feasibility(result, spec) == []

    def test_soc_stays_within_the_declared_band(self, spec):
        rng = np.random.default_rng(1)
        prices = list(rng.normal(80, 90, size=48))
        result = solve_dispatch(prices, spec)
        assert min(result.soc_mwh) >= spec.soc_min_mwh - 1e-4
        assert max(result.soc_mwh) <= spec.soc_max_mwh + 1e-4

    def test_starting_soc_is_respected(self, spec):
        result = solve_dispatch([10.0, 200.0], spec, initial_soc_mwh=90.0)
        expected = 90.0 + spec.eta_charge * result.charge_mw[0] - (
            result.discharge_mw[0] / spec.eta_discharge
        )
        assert result.soc_mwh[0] == pytest.approx(expected, abs=1e-4)

    def test_starting_soc_outside_the_band_is_rejected(self, spec):
        with pytest.raises(ValueError):
            solve_dispatch([50.0], spec, initial_soc_mwh=99.0)

    def test_empty_price_series_is_rejected(self, spec):
        with pytest.raises(ValueError):
            solve_dispatch([], spec)


class TestComplementarity:
    """The binary that stops the model inventing money at negative prices."""

    def test_constrained_solution_never_does_both_at_once(self, spec):
        prices = [-100.0] * 12 + [50.0] * 12
        result = solve_dispatch(prices, spec, enforce_complementarity=True)
        assert not any(
            c > 1e-6 and d > 1e-6
            for c, d in zip(result.charge_mw, result.discharge_mw, strict=True)
        )

    def test_the_relaxation_is_not_free_at_negative_prices(self, spec):
        """The relaxed model must earn strictly more, or the constraint is idle.

        If this ever stops holding, the argument for paying for integer
        variables no longer applies and the model should be simplified.
        """
        prices = [-100.0] * 12 + [50.0] * 12
        strict = solve_dispatch(prices, spec, enforce_complementarity=True)
        relaxed = solve_dispatch(prices, spec, enforce_complementarity=False)
        assert relaxed.net_revenue_eur(spec) > strict.net_revenue_eur(spec) + 1e-6

    def test_the_relaxation_is_harmless_at_positive_prices(self, spec):
        """With positive prices the textbook claim does hold, and we show it."""
        prices = [20.0] * 12 + [200.0] * 12
        strict = solve_dispatch(prices, spec, enforce_complementarity=True)
        relaxed = solve_dispatch(prices, spec, enforce_complementarity=False)
        assert relaxed.net_revenue_eur(spec) == pytest.approx(
            strict.net_revenue_eur(spec), rel=1e-6
        )


class TestPolicySimulator:
    def test_requests_beyond_rated_power_are_clipped(self, spec):
        result = apply_schedule([1000.0], spec, [10.0])
        assert result.charge_mw[0] == pytest.approx(spec.power_mw)

    def test_charging_stops_at_the_upper_soc_limit(self, spec):
        n = 24
        result = apply_schedule([spec.power_mw] * n, spec, [10.0] * n)
        assert max(result.soc_mwh) <= spec.soc_max_mwh + 1e-6
        assert check_feasibility(result, spec) == []

    def test_discharging_stops_at_the_lower_soc_limit(self, spec):
        n = 24
        result = apply_schedule([-spec.power_mw] * n, spec, [200.0] * n)
        assert min(result.soc_mwh) >= spec.soc_min_mwh - 1e-6
        assert check_feasibility(result, spec) == []

    def test_a_zero_request_does_nothing(self, spec):
        result = apply_schedule([0.0] * 5, spec, [50.0] * 5)
        assert result.energy_charged_mwh == 0.0
        assert result.energy_discharged_mwh == 0.0

    def test_the_simulator_never_charges_and_discharges_at_once(self, spec):
        rng = np.random.default_rng(2)
        requests = list(rng.uniform(-30, 30, size=100))
        result = apply_schedule(requests, spec, [50.0] * 100)
        assert check_feasibility(result, spec) == []

    def test_optimiser_beats_every_heuristic_on_the_same_prices(self, spec):
        """Sanity: if a rule beats the optimum, the optimum is wrong."""
        rng = np.random.default_rng(3)
        prices = list(rng.normal(80, 70, size=24 * 14))
        hours = [h % 24 for h in range(len(prices))]

        optimal = solve_dispatch(prices, spec).net_revenue_eur(spec)
        fixed = apply_schedule(
            FixedSchedulePolicy().plan(prices, hours, spec), spec, prices
        ).net_revenue_eur(spec)
        threshold = apply_schedule(
            ThresholdPolicy().plan(prices, hours, spec), spec, prices
        ).net_revenue_eur(spec)

        assert optimal >= fixed - 1e-6
        assert optimal >= threshold - 1e-6


class TestPolicies:
    def test_overlapping_charge_and_discharge_hours_are_rejected(self):
        with pytest.raises(ValueError):
            FixedSchedulePolicy(charge_hours=(1, 2), discharge_hours=(2, 3))

    def test_threshold_quantiles_must_be_ordered(self):
        with pytest.raises(ValueError):
            ThresholdPolicy(low_quantile=0.8, high_quantile=0.2)

    def test_threshold_policy_uses_no_future_information(self, spec):
        """Changing a future price must not change the action taken now.

        This is the property that makes the policy implementable rather than
        clairvoyant, and it is easy to break by an off-by-one in the window.
        """
        prices = list(np.linspace(10, 200, 200))
        hours = [h % 24 for h in range(len(prices))]
        policy = ThresholdPolicy()

        baseline = policy.plan(prices, hours, spec)

        tampered = list(prices)
        tampered[150:] = [9999.0] * 50
        altered = policy.plan(tampered, hours, spec)

        assert baseline[:150] == altered[:150]


class TestThePerfectForesightBenchmarkIsNotAnUpperBound:
    """What `perfect_foresight_dispatch` computes, and what it does not.

    Its docstring used to say "no real policy can beat this" one paragraph
    above saying the fortnight chunking "can only ever understate the optimum".
    Both cannot hold. `scripts/decompose_gap.py` found the policy that settles
    it -- a rolling 48-hour horizon on realised prices earns EUR 2,574,513
    against the benchmark's EUR 2,573,659 over 2024, because a sliding window
    sees across seams that fixed chunks cannot.

    The margin is 0.03% and changes no conclusion. It changes a word: capture
    rates here are fractions of a chunked benchmark, not of the maximum
    attainable. These tests keep that property from drifting back into a
    stronger claim.
    """

    def test_chunking_can_only_lose_revenue(self, spec):
        """The approximation has a direction, and this is it.

        One series, solved whole and solved in chunks. The chunked answer can
        never be the larger one, because every schedule it can produce is also
        available to the unchunked problem.
        """
        rng = np.random.default_rng(7)
        prices = list(rng.normal(60, 40, size=96))

        whole = solve_dispatch(prices, spec)
        chunked = perfect_foresight_dispatch(prices, spec, chunk_hours=24)

        assert chunked.net_revenue_eur(spec) <= whole.net_revenue_eur(spec) + 1e-6

    def test_a_finer_chunk_is_never_worth_more_than_a_coarser_one(self, spec):
        """More seams cannot help, so the benchmark is monotone in chunk size."""
        rng = np.random.default_rng(11)
        prices = list(rng.normal(60, 40, size=96))

        coarse = perfect_foresight_dispatch(prices, spec, chunk_hours=48)
        fine = perfect_foresight_dispatch(prices, spec, chunk_hours=12)

        assert fine.net_revenue_eur(spec) <= coarse.net_revenue_eur(spec) + 1e-6

    def test_the_benchmark_is_chunked_at_all(self, spec):
        """If it ever becomes a single solve, the caveat above stops applying.

        A long enough series with a price pattern that rewards carrying energy
        across a chunk boundary must score strictly below the unchunked answer.
        Cheap for two weeks, expensive for two weeks: an unchunked optimiser
        fills up in the first fortnight and sells in the second, and a chunked
        one cannot.
        """
        prices = [10.0] * 48 + [500.0] * 48
        whole = solve_dispatch(prices, spec)
        chunked = perfect_foresight_dispatch(prices, spec, chunk_hours=48)

        assert chunked.net_revenue_eur(spec) < whole.net_revenue_eur(spec), (
            "the benchmark appears not to be chunked any more; "
            "docs/DECISIONS.md section 5.4 assumes it is"
        )
