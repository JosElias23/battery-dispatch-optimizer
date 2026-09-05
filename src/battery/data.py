"""Day-ahead electricity price data.

Source: the Fraunhofer ISE energy-charts API, which republishes German
day-ahead auction results from the Bundesnetzagentur (SMARD.de) under
CC BY 4.0. It needs no API key and no registration, so anyone who clones this
repository can regenerate every number in it.

Why Germany and not Chile
-------------------------
Chile's Coordinador Electrico Nacional publishes marginal costs through an API
that requires a registered key, and the CNE open-data portal was unreachable at
the time of writing. Requiring a credential would break the reproducibility
guarantee this project is built on.

The German market is also the better test case. Very high renewable
penetration produces extreme price volatility -- 5.2% of 2024 hours cleared
*below zero* -- and volatility is the entire economic basis for storage
arbitrage. A battery optimiser evaluated on a flat price curve demonstrates
nothing.

`fetch_prices` is deliberately the only function that touches the network, and
`PriceSeries` is a plain container, so adding a Chilean source later means
writing one adapter and changing nothing else.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from battery.utils import PROJECT_ROOT

USER_AGENT = "battery-dispatch-optimizer/0.1 (github.com/JosElias23)"


class DataError(RuntimeError):
    """Raised when the downloaded price series fails validation."""


@dataclass(frozen=True)
class PriceSeries:
    """Hourly day-ahead prices with their timestamps."""

    timestamps: list[datetime]
    prices_eur_mwh: list[float]
    bidding_zone: str
    license_info: str = ""

    def __post_init__(self) -> None:
        if len(self.timestamps) != len(self.prices_eur_mwh):
            raise DataError(
                f"{len(self.timestamps)} timestamps vs {len(self.prices_eur_mwh)} prices"
            )

    def __len__(self) -> int:
        return len(self.prices_eur_mwh)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self.prices_eur_mwh, dtype=float)

    def slice_year(self, year: int) -> PriceSeries:
        """Extract one calendar year, in the market's own local time.

        The API returns UTC timestamps. Prices are set by human demand, which
        follows local clocks, so hour-of-day features must be built on local
        time or the daily shape is smeared across the year by the summer-time
        shift. Central European Time is UTC+1, ignoring the one-hour summer
        offset; the residual error is absorbed by the calendar features.
        """
        keep = [
            (ts, price)
            for ts, price in zip(self.timestamps, self.prices_eur_mwh, strict=True)
            if (ts + timedelta(hours=1)).year == year
        ]
        if not keep:
            raise DataError(f"No data for year {year}")
        return PriceSeries(
            timestamps=[ts for ts, _ in keep],
            prices_eur_mwh=[p for _, p in keep],
            bidding_zone=self.bidding_zone,
            license_info=self.license_info,
        )

    def stats(self) -> dict:
        v = self.values
        return {
            "n_hours": len(v),
            "start": self.timestamps[0].isoformat(),
            "end": self.timestamps[-1].isoformat(),
            "mean_eur_mwh": round(float(v.mean()), 2),
            "std_eur_mwh": round(float(v.std()), 2),
            "min_eur_mwh": round(float(v.min()), 2),
            "max_eur_mwh": round(float(v.max()), 2),
            "negative_hours": int((v < 0).sum()),
            "negative_hours_pct": round(100 * float((v < 0).mean()), 2),
            "mean_daily_spread_eur_mwh": round(float(np.mean(self.daily_spreads())), 2),
        }

    def daily_spreads(self) -> np.ndarray:
        """Max minus min price within each complete day.

        This is the raw arbitrage opportunity: no round trip can earn more than
        the daily spread, before efficiency losses.
        """
        v = self.values
        usable = len(v) - (len(v) % 24)
        days = v[:usable].reshape(-1, 24)
        return days.max(axis=1) - days.min(axis=1)


def _cache_path(bidding_zone: str, start: str, end: str, raw_dir: Path) -> Path:
    return raw_dir / f"prices_{bidding_zone}_{start}_{end}.json"


def fetch_prices(
    bidding_zone: str,
    start: str,
    end: str,
    api_url: str = "https://api.energy-charts.info/price",
    raw_dir: str | Path = "data/raw",
    force: bool = False,
    timeout: int = 120,
) -> PriceSeries:
    """Download hourly day-ahead prices, caching the raw response on disk.

    The cache is keyed on the exact query, so re-running any script is offline
    and instant, and the bytes behind a published number stay on disk.
    """
    raw_path = Path(raw_dir)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    raw_path.mkdir(parents=True, exist_ok=True)

    cache = _cache_path(bidding_zone, start, end, raw_path)
    if cache.exists() and not force:
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        url = f"{api_url}?bzn={bidding_zone}&start={start}&end={end}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        cache.write_text(json.dumps(payload), encoding="utf-8")

    return _parse_payload(payload, bidding_zone)


def _parse_payload(payload: dict, bidding_zone: str) -> PriceSeries:
    if "unix_seconds" not in payload or "price" not in payload:
        raise DataError(f"Unexpected API payload, keys were {sorted(payload)}")

    seconds = payload["unix_seconds"]
    prices = payload["price"]

    # Missing hours are dropped rather than interpolated. An interpolated price
    # is a price no one traded at, and the optimiser would happily arbitrage
    # against it and book revenue that never existed.
    kept = [(s, p) for s, p in zip(seconds, prices, strict=True) if p is not None]
    dropped = len(prices) - len(kept)
    if dropped:
        print(f"[data] dropped {dropped} hours with no published price")

    if not kept:
        raise DataError("Payload contained no usable prices")

    series = PriceSeries(
        timestamps=[datetime.fromtimestamp(s, tz=timezone.utc) for s, _ in kept],
        prices_eur_mwh=[float(p) for _, p in kept],
        bidding_zone=bidding_zone,
        license_info=payload.get("license_info", ""),
    )
    validate(series)
    return series


def validate(series: PriceSeries) -> None:
    """Fail loudly on a series that would silently corrupt every downstream number."""
    if len(series) < 24:
        raise DataError(f"Only {len(series)} hours; need at least one full day")

    deltas = {
        (b - a).total_seconds()
        for a, b in zip(series.timestamps, series.timestamps[1:], strict=False)
    }
    unexpected = deltas - {3600.0}
    if unexpected:
        raise DataError(f"Series is not strictly hourly; found gaps of {sorted(unexpected)}s")

    v = series.values
    if not np.isfinite(v).all():
        raise DataError("Series contains NaN or infinite prices")

    # Sanity bound, not a business rule. German day-ahead prices have cleared
    # near -500 and above +2500 EUR/MWh in real crises; anything outside this
    # window means the units or the column are wrong.
    if v.min() < -1000 or v.max() > 5000:
        raise DataError(f"Prices outside plausible range: [{v.min()}, {v.max()}] EUR/MWh")


def load_years(config: dict, force: bool = False) -> dict[int, PriceSeries]:
    """Load the configured training and test years as separate series."""
    data_cfg = config["data"]
    years = [data_cfg["train_year"], data_cfg["test_year"]]

    # One request spanning both years, then split locally: fewer round trips,
    # and the split boundary is handled in exactly one place.
    combined = fetch_prices(
        bidding_zone=data_cfg["bidding_zone"],
        start=f"{min(years)}-01-01",
        end=f"{max(years)}-12-31",
        api_url=data_cfg["api_url"],
        raw_dir=data_cfg["raw_dir"],
        force=force,
    )
    return {year: combined.slice_year(year) for year in years}
