# Decision log

What was built, in what order, why each choice was made, and what the numbers
turned out to be. Every figure here comes from a file in `reports/`, produced by
running the code. Nothing is typed from memory.

---

## 1. The question

A 100 MWh / 25 MW battery is dispatched against a real day-ahead market. How
much of the theoretically attainable revenue can an operator capture without
knowing tomorrow's prices?

The framing matters more than it looks. "The battery earned €2.2 M" is
unfalsifiable: it depends entirely on how volatile the year happened to be. "The
battery captured 85.7% of the perfect-foresight optimum" is a claim about the
*method*, and it is comparable across markets and years.

That framing forces a decision most dispatch projects skip: the upper bound has
to be computed, not assumed. Without it there is no denominator.

---

## 2. Decisions about the data

### 2.1 German prices, not Chilean ones

Chile is the market I actually care about, and it is not the market used here.

The Coordinador Eléctrico Nacional API rejects unauthenticated requests
(`Authentication parameters missing`) and the CNE open-data portal was
unreachable throughout development. Requiring a credential would break the one
guarantee this repository makes: clone it, run it, get these numbers.

German/Luxembourg day-ahead prices come from the energy-charts API (Fraunhofer
ISE), republishing Bundesnetzagentur/SMARD data under CC BY 4.0, with no key and
no registration.

Germany is also the better test case on the merits. Very high renewable
penetration produces exactly the volatility and the negative prices that make
storage arbitrage interesting, and those negative prices turn out to drive
finding 5.3. All network access is isolated behind one function in
`src/battery/data.py`, so a Chilean adapter is one class and no other change.

### 2.2 The split is across years, not within one

2023 trains, 2024 evaluates. A random or chronological split *inside* one year
would have looked cleaner and hidden the most important failure in the project.

| | 2023 (train) | 2024 (test) |
|---|---:|---:|
| Hours | 8,760 | 8,784 |
| Mean price | 95.18 €/MWh | 78.51 €/MWh |
| Std. deviation | 47.58 | 52.72 |
| Range | −500.00 to 524.27 | −135.45 to 936.28 |
| Negative-price hours | 301 (3.44%) | **457 (5.20%)** |
| Mean daily spread | 98.13 | 111.47 |

The mean price fell by 16.67 €/MWh between the two years. That is the
distribution shift that broke the first forecaster (5.2), and it is only visible
because the split spans years, which is also the only split that resembles
deployment.

### 2.3 No feature may see within 48 hours of its target

The rolling horizon plans 48 hours ahead, so the furthest hour being decided is
48 hours away. Any feature referencing a price closer than that is information
the operator would not have had.

This rules out the single strongest predictor available, yesterday's price at
the same hour. `src/battery/forecast.py` raises on any lag shorter than 48, and
that guard immediately caught a lag of 24 left in the config from an early
draft. It would have produced a better forecast, a better capture rate, and
revenue no one could ever have earned.

`tests/test_forecast.py` goes further than an assertion on the config: it
overwrites the second half of the price series with garbage and requires every
prediction in the first half to be bit-for-bit unchanged. Any feature reaching
forward in time fails it, including one added later by someone who has not read
this file.

---

## 3. Decisions about the physics

### 3.1 Three costs, all of which lower the answer

- **Round-trip efficiency 0.86.** Delivering 1 MWh to the grid requires drawing
  1.163 MWh from it. Buying at 100 €/MWh means the sell price must clear 119.28
  €/MWh, degradation included, just to break even.
- **Usable band 10–90% of state of charge**, so 80 of the 100 MWh are available.
  Cycling a cell to either extreme is what actually destroys it.
- **Degradation 3 €/MWh of throughput.** Without a cost on throughput the
  optimiser trades every small wiggle: profitable on paper, and it wears the
  asset out.

Each of these is a modelling choice that reduces the reported revenue. Omitting
any one of them produces a bigger, less defensible number.

### 3.2 The solver's answer is re-checked by code that is not the solver

`check_feasibility` re-derives the state of charge hour by hour from a returned
schedule and re-tests every physical limit independently.

A solver reporting `Optimal` means only that it satisfied the constraints it was
handed. If those were written incorrectly, the answer is optimal for the wrong
problem, and the mistake surfaces as *revenue* rather than as an error, which is
the worst possible failure mode for this project. Every dispatch in the results
table passes the independent check before its revenue is recorded.

### 3.3 Plan on forecasts, book at realised prices

The rolling horizon plans on forecast prices and commits the first 24 hours of a
48-hour plan. Revenue is always accounted at the prices that actually cleared.

Scoring a forecast-driven plan against the forecast that produced it measures
nothing except the optimiser's arithmetic, and would have produced a capture
rate near 100%.

Committing less than the planning horizon is what stops the schedule collapsing
at the window edge: energy left in the battery at the end of a window is worth
nothing *inside* that window, so a model that commits everything it plans
empties the battery every night regardless of tomorrow. The discarded second day
absorbs the end effect.

---

## 4. Results

Net revenue over 2024 (8,784 hours), after round-trip losses and degradation.

| Policy | Net revenue | Capture rate | Cycles/day |
|---|---:|---:|---:|
| Fixed schedule (charge at night, sell at peak) | € 1,015,207 | 39.5% | 0.93 |
| Price threshold (quantile rule) | € 951,999 | 37.0% | 0.67 |
| MILP + seasonal-naive forecast | € 2,154,597 | 83.7% | 1.42 |
| **MILP + gradient-boosted forecast** | **€ 2,205,860** | **85.7%** | 1.50 |
| _Perfect foresight (bound)_ | _€ 2,573,659_ | _100%_ | 1.45 |

Optimisation against a forecast captures 85.7% of what a clairvoyant operator
could have earned, and earns € 1.19 M more than a fixed schedule, 2.17× the
revenue from the same physical asset.

The threshold rule is worth a second look, because it is the policy most people
reach for first and it performs *worse* than the naive fixed schedule. It cycles
only 0.67 times a day: a static quantile cutoff fitted on one year sits in the
wrong place in the next, so the rule sits out opportunities it should have
taken. That is the same level-versus-shape failure as 5.2, arriving in a
rule-based policy instead of a learned one.

---

## 5. The five decisions that changed the result

### 5.1 A better forecast is not automatically worth more money

| Forecaster | MAE | RMSE | Revenue | Capture |
|---|---:|---:|---:|---:|
| Seasonal naive (same hour, last week) | 32.31 | 54.78 | € 2,154,597 | 83.7% |
| Gradient boosting | **28.36** | **43.94** | **€ 2,205,860** | **85.7%** |

A 12% reduction in MAE buys 2.0 percentage points of capture. Real, and far from
proportional.

Dispatch does not need accurate prices; it needs the *ordering* of cheap and
expensive hours. Getting the level of a flat afternoon wrong costs nothing.
Getting the position of the evening peak wrong costs a full cycle.

An intermediate result made the point harder to argue with. An earlier booster
had a *worse* MAE than the naive baseline (25.97 against 24.77 on a two-month
slice), and the fix that mattered lifted capture from 44.5% to 55.1% on that
same slice while barely moving MAE. **Forecast error is a proxy metric; revenue
is the objective.** Both are reported here, and they disagree.

### 5.2 Training on price levels broke the model; centring fixed it

The first gradient booster predicted the price directly. It learned 2023, where
prices averaged 95.18 €/MWh, and was evaluated on 2024, which cleared at 78.51.
Every notion of "cheap" it held sat 17 €/MWh too high, and it lost to a weekly
lag.

The fix: predict the *deviation from a trailing 168-hour level*, with the lagged
features centred on the same anchor. The daily and weekly **shape** of
electricity prices is stable across years. The absolute level is not, and the
level is what moved.

This is the decision that most changed the outcome, and it is a data decision,
not a modelling one. No amount of hyperparameter search on the original target
would have recovered it.

### 5.3 The constraint everyone calls redundant is not, but it is cheap

Textbook formulations often drop the binary forbidding simultaneous charge and
discharge, arguing that doing both at once is never optimal, so the linear
relaxation is exact and much faster.

That argument assumes positive prices. In 457 hours of 2024, 5.2% of the year,
the price cleared below zero, and there the operator is *paid* to consume. The
relaxed model then discovers it can hold the state of charge flat while
maintaining a net import:

```
hold SoC:   eta_c * charge = discharge / eta_d
net grid:   discharge - charge = charge * (eta_rt - 1) < 0
```

The round-trip loss becomes a way to absorb paid-for energy indefinitely,
without ever filling the battery. Running `scripts/ablate_complementarity.py`:

| | With constraint | Without |
|---|---:|---:|
| Net revenue | € 2,573,658.64 | € 2,574,867.31 |
| Solve time | 4.43 s | 2.80 s |
| Hours charging and discharging at once | 0 | **28** |
| ...of which at negative prices | — | **28 (100%)** |

The most extreme case charges at 25 MW while discharging 11.8 MW at −85.08
€/MWh.

**The honest conclusion is about correctness, not money.** Every one of the 28
impossible hours occurs at a negative price, exactly as the mechanism predicts,
which is strong evidence the mechanism is understood rather than coincidental.
But the invented revenue is € 1,208.67, or **0.047%**. The relaxation produces a
physically impossible schedule and is barely richer for it.

Nor is it the dramatic speed-up the textbook argument implies: 4.43 s against
2.80 s for the full year, a factor of 1.6. Modern branch-and-bound handles this
structure well.

Keep the binary. It costs 1.6 seconds a year and it is the difference between a
schedule an asset could execute and one it could not. Writing this up as "the
constraint is worth € 1,209 a year" would have been a much more sellable claim
than the one the numbers actually support.

### 5.4 The remaining 14% is a forecasting problem, not an optimisation one

The MILP is optimal for the prices it is given. The 14.3% gap to perfect
foresight is therefore attributable to forecast error, every euro of it. No
improvement to the solver, the formulation or the horizon can recover any of it.

That is a useful thing to know *before* spending effort on the solver, and it is
only knowable because the upper bound was computed.

### 5.5 Revenue is insensitive to when forecast errors land

300 simulated years, each built by resampling the observed 2024 forecast errors
in day-long blocks and re-running the full rolling-horizon dispatch:

| | |
|---|---:|
| Mean annual revenue | € 1,979,567.41 |
| Standard deviation | € 24,533.02 |
| 95% interval | € 1,923,083.17 – € 2,025,870.43 |
| 5th / 95th percentile | € 1,938,540.21 / € 2,016,983.82 |
| **Relative spread (1 sd)** | **1.24% of mean** |

Given a forecaster of this quality, the *timing* of its mistakes barely matters:
bad luck costs about € 25,000 on € 2 M. For an asset owner that is the
difference between a revenue model worth underwriting and one that is not.

**The realised run beat the simulation, and that needed explaining rather than
celebrating.** The gradient-boosting dispatch earned € 2,205,860, above the 95%
interval. Either it was lucky, or the simulation is pessimistic by construction.
`scripts/analyse_forecast_error.py` tests the second hypothesis:

| Forecast | MAE | Mean within-day rank correlation |
|---|---:|---:|
| Real gradient boosting | 28.360 | **0.7757** |
| Block-bootstrapped | 28.225 | **0.7266** (sd 0.0119) |

Same error magnitude, materially worse ranking: 0.049 of rank fidelity lost.
Block resampling preserves how *big* errors are and how they cluster in time,
but detaches them from the prices they were made against. An error block from a
volatile December week pasted onto a calm July day is a forecast no model would
have produced.

And dispatch value lives in the ranking: a forecast uniformly 30 €/MWh too high
loses nothing, because the optimiser still buys in the same hours. So the Monte
Carlo is a **conservative bound**, and the realised result is structure the
bootstrap discards, not luck.

This is 5.1 arriving by a second, independent route. Two experiments, one
conclusion: for storage dispatch, ordering is the metric and error magnitude is
a decoy.

One complication surfaced along the way, and it cuts against the project. **The
forecaster is worst exactly where the money is:** MAE 27.63 in the calmest
quartile of hours against 34.18 in the most volatile, a ratio of 1.237, with
correlation 0.355 between error and local volatility. Volatile hours are where
the spreads are. Improving the forecast specifically there is the most promising
route to closing the remaining 14%.

---

## 6. What was not done

- **One year, one market, one battery.** 2024 Germany, 4-hour duration. Capture
  rates depend on the year's volatility, the asset's duration and the market's
  structure. Nothing here is tested on Chilean data.
- **The 48-hour blackout is conservative.** The German auction clears around
  12:45 on D−1 and publishes all 24 prices for day D, so an operator planning
  *within* the delivery day has near-perfect foresight. The problem modelled
  here is the one faced when bidding *into* the auction; a desk bidding at noon
  on D−1 faces 12 to 36 hours, not 48. A shorter blackout would raise every
  capture rate above. It would not change the ranking.
- **Arbitrage only.** Real storage earns a large share of its income from
  frequency response and capacity markets. These figures are a lower bound on
  asset value and are not a business case.
- **Degradation is linear in throughput.** Real ageing depends on depth of
  discharge, temperature, C-rate and calendar time. The linear approximation
  keeps the problem MILP-representable and will misprice deep cycles.
- **The battery starts half full**, so 50 MWh can be sold without having been
  bought. Worth roughly 0.1% of annual revenue and applied identically to every
  policy, so comparisons are unaffected and absolute figures are very slightly
  flattered.
- **Perfect foresight is solved in fortnight chunks** chained through the state
  of charge, not as one 8,784-hour MILP. This can only *understate* the true
  optimum, since it forbids arbitrage across chunk boundaries, so the reported
  capture rates are if anything slightly generous.
- **No hyperparameter search.** Standard booster settings, untuned. Fair as a
  comparison, almost certainly not optimal.
- **The Monte Carlo models error timing, not error size.** It resamples the
  errors the fitted model actually made, answering "what if these mistakes had
  fallen elsewhere in the year", not "what if the forecaster were worse". The
  second question needs a magnitude sensitivity and is not answered here.

---

## 7. Reproducing

```bash
pip install -e ".[dev]"
python -m pytest                                 # 66 tests
python scripts/run_experiment.py                 # all policies, about a minute
python scripts/ablate_complementarity.py         # the money-pump ablation
python scripts/analyse_forecast_error.py         # why the Monte Carlo is conservative
python scripts/run_monte_carlo.py --workers 12   # about an hour, 300 x 366 solves
python scripts/make_figures.py
```

Every number in this document lives in `reports/metrics_dispatch.json`,
`reports/metrics_complementarity_ablation.json`, `reports/metrics_monte_carlo.json` and
`reports/metrics_forecast_error_analysis.json`.
