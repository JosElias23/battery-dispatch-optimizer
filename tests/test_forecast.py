"""Tests for the forecasters, centred on temporal leakage.

Leakage is the failure mode that matters here. A forecaster that peeks at a
price it could not have known produces a better MAE, a better-looking dispatch,
and revenue that could never have been earned. Nothing crashes, so only a test
catches it.
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from battery.forecast import (
    MIN_LAG_HOURS,
    GradientBoostingForecaster,
    SeasonalNaiveForecaster,
    calendar_features,
    evaluate_forecast,
    lag_features,
)


def hourly_timestamps(n: int, start: datetime | None = None) -> list[datetime]:
    start = start or datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(hours=i) for i in range(n)]


@pytest.fixture
def synthetic_prices():
    """A daily sine plus a weekly component and noise: realistic in shape."""
    rng = np.random.default_rng(0)
    t = np.arange(24 * 60)
    return (
        80.0
        + 30 * np.sin(2 * np.pi * t / 24)
        + 10 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 5, size=len(t))
    )


class TestLeakageGuards:
    def test_a_lag_inside_the_blackout_is_rejected(self):
        with pytest.raises(ValueError, match="blackout"):
            lag_features(np.arange(500.0), lags=[24], rolling_windows=[24])

    @pytest.mark.parametrize("lag", [1, 12, 23, 47])
    def test_every_short_lag_is_rejected(self, lag):
        with pytest.raises(ValueError):
            lag_features(np.arange(500.0), lags=[lag], rolling_windows=[24])

    def test_lags_at_or_beyond_the_blackout_are_accepted(self):
        _matrix, valid = lag_features(
            np.arange(1000.0), lags=[MIN_LAG_HOURS, 72], rolling_windows=[24]
        )
        assert valid.any()

    def test_lag_columns_contain_the_correct_past_value(self):
        prices = np.arange(500.0)
        matrix, _ = lag_features(prices, lags=[48], rolling_windows=[])
        # Row t must hold the price from t-48.
        assert matrix[100, 0] == pytest.approx(prices[52])
        assert matrix[300, 0] == pytest.approx(prices[252])

    def test_rolling_windows_end_before_the_blackout(self):
        prices = np.arange(1000.0)
        matrix, _ = lag_features(prices, lags=[48], rolling_windows=[24])
        # Column 1 is the rolling mean over the 24 hours ending at t-48.
        t = 500
        expected = prices[t - MIN_LAG_HOURS - 24 : t - MIN_LAG_HOURS].mean()
        assert matrix[t, 1] == pytest.approx(expected)

    def test_future_prices_cannot_change_a_past_prediction(self, synthetic_prices):
        """The decisive test: tamper with the future, the past must not move.

        If any feature or the level anchor reaches forward in time, predictions
        before the tampering point will shift.
        """
        timestamps = hourly_timestamps(len(synthetic_prices))
        model = GradientBoostingForecaster(n_estimators=30).fit(
            synthetic_prices, timestamps
        )
        baseline = model.predict(synthetic_prices, timestamps)

        tampered = synthetic_prices.copy()
        cut = len(tampered) // 2
        tampered[cut:] = 10_000.0
        altered = model.predict(tampered, timestamps)

        np.testing.assert_allclose(baseline[:cut], altered[:cut], rtol=1e-9)

    def test_the_naive_forecaster_only_looks_backward(self, synthetic_prices):
        timestamps = hourly_timestamps(len(synthetic_prices))
        model = SeasonalNaiveForecaster()
        baseline = model.predict(synthetic_prices, timestamps)

        tampered = synthetic_prices.copy()
        cut = len(tampered) // 2
        tampered[cut:] = 10_000.0
        altered = model.predict(tampered, timestamps)

        np.testing.assert_allclose(baseline[:cut], altered[:cut], rtol=1e-9)


class TestForecasterBehaviour:
    def test_predictions_are_complete_and_finite(self, synthetic_prices):
        """The dispatch model cannot consume a NaN, so none may escape."""
        timestamps = hourly_timestamps(len(synthetic_prices))
        model = GradientBoostingForecaster(n_estimators=30).fit(
            synthetic_prices, timestamps
        )
        predictions = model.predict(synthetic_prices, timestamps)
        assert len(predictions) == len(synthetic_prices)
        assert np.isfinite(predictions).all()

    def test_naive_forecaster_repeats_the_previous_week(self, synthetic_prices):
        timestamps = hourly_timestamps(len(synthetic_prices))
        predictions = SeasonalNaiveForecaster().predict(synthetic_prices, timestamps)
        assert predictions[200] == pytest.approx(synthetic_prices[200 - 168])

    def test_the_booster_beats_the_naive_baseline_on_seasonal_data(self, synthetic_prices):
        """On data with a clean daily shape, the model must add value.

        This is not a claim about real prices, which are measured on the actual
        market data and reported honestly in the README. But if the model
        cannot beat a naive lag on a sine wave, something is wired wrong.
        """
        timestamps = hourly_timestamps(len(synthetic_prices))
        split = len(synthetic_prices) // 2

        model = GradientBoostingForecaster(n_estimators=200).fit(
            synthetic_prices[:split], timestamps[:split]
        )
        gbm = model.predict(synthetic_prices, timestamps)[split:]
        naive = SeasonalNaiveForecaster().predict(synthetic_prices, timestamps)[split:]
        actual = synthetic_prices[split:]

        assert np.abs(gbm - actual).mean() < np.abs(naive - actual).mean()

    def test_level_shift_does_not_break_the_forecaster(self, synthetic_prices):
        """The failure this design exists to prevent.

        Training at one price level and predicting at another must not destroy
        accuracy. The first version of this model learned 2023 levels and fell
        apart on 2024, which cleared 17 EUR/MWh lower.
        """
        timestamps = hourly_timestamps(len(synthetic_prices))
        split = len(synthetic_prices) // 2

        shifted = synthetic_prices.copy()
        shifted[split:] -= 40.0  # a large, abrupt regime change

        model = GradientBoostingForecaster(n_estimators=200).fit(
            shifted[:split], timestamps[:split]
        )
        predictions = model.predict(shifted, timestamps)[split + 336 :]
        actual = shifted[split + 336 :]

        # Bias must stay small relative to the 40 EUR/MWh shift itself.
        assert abs(np.mean(predictions - actual)) < 10.0


class TestCalendarFeatures:
    def test_shape_is_stable(self):
        assert calendar_features(hourly_timestamps(50)).shape == (50, 8)

    def test_hour_is_local_not_utc(self):
        # 23:00 UTC is midnight in Central European Time.
        features = calendar_features([datetime(2024, 6, 1, 23, tzinfo=timezone.utc)])
        assert features[0, 0] == 0

    def test_cyclical_encoding_makes_hour_23_adjacent_to_hour_0(self):
        features = calendar_features(hourly_timestamps(48))
        # Compare the sin/cos pair rather than the raw integer.
        near = np.linalg.norm(features[23, 1:3] - features[24, 1:3])
        far = np.linalg.norm(features[0, 1:3] - features[12, 1:3])
        assert near < far


class TestEvaluation:
    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ValueError):
            evaluate_forecast(np.zeros(5), np.zeros(6), "x")

    def test_nan_predictions_are_rejected(self):
        with pytest.raises(ValueError):
            evaluate_forecast(np.array([1.0, np.nan]), np.zeros(2), "x")

    def test_a_perfect_forecast_scores_zero_error(self):
        actual = np.array([10.0, 20.0, 30.0])
        result = evaluate_forecast(actual.copy(), actual, "perfect")
        assert result.mae == 0.0
        assert result.rmse == 0.0
