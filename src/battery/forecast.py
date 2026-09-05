"""Day-ahead price forecasting.

The optimiser needs prices it does not yet know. This module supplies them, and
the constraint that shapes every design choice here is *what was knowable when*.

The decision timing
-------------------
The battery re-plans once a day and commits 24 hours, planning over a 48-hour
horizon. At the moment of planning, the furthest hour being decided is 48 hours
away. Every feature must therefore be available at least 48 hours before the
hour it predicts.

That rules out the obvious and much stronger feature, yesterday's price at the
same hour (`lag 24`), which is why the minimum lag here is 48 and not 24. Using
lag 24 would cut the forecast error substantially and the resulting revenue
figure would be fiction: the model would be trading on a price it could not
have seen. `tests/test_forecast.py` asserts that no feature draws on anything
inside the 48-hour blackout.

Forecast quality is reported two ways: as error (MAE, RMSE) and as the revenue
the dispatch actually earns using it. They are not the same thing, and the
second is the one that matters. A forecast can be more accurate on average and
still be worth less money, because dispatch only cares about getting the
*ordering* of cheap and expensive hours right.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# No feature may reference an hour closer than this to its target.
MIN_LAG_HOURS = 48


@dataclass
class ForecastResult:
    """Predicted prices aligned with the truth, plus error metrics."""

    predictions: np.ndarray
    actuals: np.ndarray
    name: str

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.predictions - self.actuals)))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(np.mean((self.predictions - self.actuals) ** 2)))

    @property
    def errors(self) -> np.ndarray:
        return self.predictions - self.actuals

    def metrics(self) -> dict:
        return {
            "name": self.name,
            "mae_eur_mwh": round(self.mae, 3),
            "rmse_eur_mwh": round(self.rmse, 3),
            "bias_eur_mwh": round(float(np.mean(self.errors)), 3),
            "n_hours": len(self.actuals),
        }


def calendar_features(timestamps: Sequence[datetime], utc_offset_hours: int = 1) -> np.ndarray:
    """Hour, weekday and seasonal position, encoded for a tree model.

    Hour and month are given as sine/cosine pairs as well as raw integers. The
    cyclical encoding stops the model treating hour 23 and hour 0 as maximally
    distant, which matters because the overnight price trough straddles
    midnight.
    """
    rows = []
    for ts in timestamps:
        local_hour = (ts.hour + utc_offset_hours) % 24
        rows.append([
            local_hour,
            np.sin(2 * np.pi * local_hour / 24),
            np.cos(2 * np.pi * local_hour / 24),
            ts.weekday(),
            1.0 if ts.weekday() >= 5 else 0.0,
            ts.month,
            np.sin(2 * np.pi * ts.month / 12),
            np.cos(2 * np.pi * ts.month / 12),
        ])
    return np.asarray(rows, dtype=float)


def lag_features(
    prices: np.ndarray,
    lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build lagged and rolling features, returning the matrix and a validity mask.

    Every lag is checked against MIN_LAG_HOURS, and rolling windows are shifted
    so they end MIN_LAG_HOURS before the target hour. Rows without full history
    are masked out rather than imputed: a zero-filled lag is a made-up price.
    """
    for lag in lags:
        if lag < MIN_LAG_HOURS:
            raise ValueError(
                f"Lag {lag} is inside the {MIN_LAG_HOURS}h blackout; it would leak "
                "information the planner could not have had."
            )

    n = len(prices)
    columns: list[np.ndarray] = []

    for lag in lags:
        column = np.full(n, np.nan)
        column[lag:] = prices[: n - lag]
        columns.append(column)

    for window in rolling_windows:
        mean_col = np.full(n, np.nan)
        std_col = np.full(n, np.nan)
        for t in range(MIN_LAG_HOURS + window, n):
            history = prices[t - MIN_LAG_HOURS - window : t - MIN_LAG_HOURS]
            mean_col[t] = history.mean()
            std_col[t] = history.std()
        columns.append(mean_col)
        columns.append(std_col)

    matrix = np.column_stack(columns)
    valid = ~np.isnan(matrix).any(axis=1)
    return matrix, valid


class SeasonalNaiveForecaster:
    """Predict each hour with the price of the same hour one week earlier.

    Electricity prices have a strong weekly cycle -- weekends are cheap because
    industrial demand falls -- so a lag of 168 hours is the standard naive
    benchmark in this literature. It requires no fitting and is often
    embarrassingly hard to beat.
    """

    name = "seasonal_naive_168h"
    lag = 168

    def fit(self, *_args, **_kwargs) -> SeasonalNaiveForecaster:
        return self

    def predict(self, prices: np.ndarray, timestamps: Sequence[datetime]) -> np.ndarray:
        predictions = np.full(len(prices), np.nan)
        predictions[self.lag :] = prices[: len(prices) - self.lag]
        # The first week has no history; fall back to the mean of what is known.
        predictions[: self.lag] = np.nanmean(prices[: self.lag])
        return predictions


class GradientBoostingForecaster:
    """Gradient-boosted trees over calendar and lagged-price features.

    Histogram-based boosting is used because it handles the ~17k training rows
    in a second or two and needs no feature scaling. The point of this model is
    not to be state of the art -- published day-ahead forecasters use weather
    and load forecasts, which are not available here -- but to be a credible,
    honestly-validated step above the naive benchmark.
    """

    name = "gradient_boosting"

    def __init__(
        self,
        lags: Sequence[int] = (48, 49, 50, 72, 96, 168, 169, 336),
        rolling_windows: Sequence[int] = (24, 168),
        n_estimators: int = 400,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        seed: int = 42,
        level_window: int = 168,
    ) -> None:
        self.lags = tuple(lags)
        self.rolling_windows = tuple(rolling_windows)
        self.level_window = level_window
        self.model = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=seed,
            early_stopping=False,
        )

    def _reference_level(self, prices: np.ndarray) -> np.ndarray:
        """Trailing mean price, ending before the blackout starts.

        This is the anchor the model predicts *relative to*. It uses only data
        older than MIN_LAG_HOURS, so it introduces no leakage.
        """
        n = len(prices)
        level = np.full(n, np.nan)
        window = self.level_window
        for t in range(MIN_LAG_HOURS + window, n):
            level[t] = prices[t - MIN_LAG_HOURS - window : t - MIN_LAG_HOURS].mean()
        return level

    def _design_matrix(
        self, prices: np.ndarray, timestamps: Sequence[datetime]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lags, valid = lag_features(prices, self.lags, self.rolling_windows)
        calendar = calendar_features(timestamps)
        level = self._reference_level(prices)
        valid = valid & ~np.isnan(level)

        # Lagged prices are centred on the same anchor as the target. Handing
        # the model absolute price levels is what tied the first version of it
        # to 2023: it learned that "cheap" meant roughly 60 EUR/MWh, and 2024
        # cleared 17 EUR/MWh lower on average, so every threshold it had
        # learned was in the wrong place.
        centred_lags = lags - level[:, None]
        features = np.column_stack([calendar, centred_lags, level])
        return features, valid, level

    def fit(
        self, prices: np.ndarray, timestamps: Sequence[datetime]
    ) -> GradientBoostingForecaster:
        features, valid, level = self._design_matrix(prices, timestamps)
        # Predict the deviation from the trailing level, not the level itself.
        # The daily and weekly *shape* of prices is stable across years; the
        # absolute level is not, and it is the level that shifted.
        target = prices[valid] - level[valid]
        self.model.fit(features[valid], target)
        return self

    def predict(self, prices: np.ndarray, timestamps: Sequence[datetime]) -> np.ndarray:
        features, valid, level = self._design_matrix(prices, timestamps)
        predictions = np.full(len(prices), np.nan)
        predictions[valid] = self.model.predict(features[valid]) + level[valid]
        # Rows without full history fall back to the naive forecast rather than
        # to a constant, so the dispatch simulation never sees a NaN.
        if (~valid).any():
            fallback = SeasonalNaiveForecaster().predict(prices, timestamps)
            predictions[~valid] = fallback[~valid]
        return predictions


def evaluate_forecast(
    predictions: np.ndarray, actuals: np.ndarray, name: str
) -> ForecastResult:
    if len(predictions) != len(actuals):
        raise ValueError(f"{len(predictions)} predictions vs {len(actuals)} actuals")
    if np.isnan(predictions).any():
        raise ValueError("Predictions contain NaN; the dispatch model cannot consume them")
    return ForecastResult(predictions=predictions, actuals=actuals, name=name)
