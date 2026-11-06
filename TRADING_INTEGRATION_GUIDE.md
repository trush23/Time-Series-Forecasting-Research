# Reading TS10: forecasts, delayed execution and trading evidence

TS10 connects the library to the **existing** NB9/NB10 timing and decision rules.
It does not replace those notebooks, retune their rules or create orders. All
results refer to the previously examined adjusted SPY snapshot ending 2026-10-01.

## The timing distinction

A TS five-step forecast issued after close t covers outcomes through t+5. NB9/NB10
enter at close t+1 and hold h close-to-close intervals, exiting at t+1+h.

| Holding horizon after entry | Last forecast step needed | Compatible with TS five-step budget? |
|---:|---:|---|
| 1 | 2 | Yes |
| 2 | 3 | Yes |
| 3, primary | 4 | Yes |
| 4 | 5 | Yes |
| 5 | 6 | Legacy reference only |

If R[j] is a sampled cumulative return from t to t+j, the delayed holding return
is **R[h+1] - R[1]**. Apply this to each joint sample before computing its quantiles.
Subtracting two marginal quantiles gives the wrong difference distribution in
general. The notebook includes an exact counterexample.

For the smoothed track, the adapter forecasts the **change** in the EMA between
entry and exit. This is a derived target, not a relabeled native smoothed-level
forecast. Only actual-return distributions feed trading decisions.

## What was reproduced

The replay restores NB9's frozen coordinates, metric, k=80, temporal exclusion,
neighbor spacing, weights and support rule. All **9,810** original forecast rows
are checked. Means, quantiles, probabilities and actual targets match exactly in
the validated environment; maximum CRPS discrepancy is about 8.3e-17.

The original NB10 analog, recent, buy-and-hold and cash ledgers reproduce within
1.4e-9 dollars, along with original metrics, cost sensitivity and every original
horizon/offset result. Its h=5 reference remains explicitly a six-step forecast.
Source exports and notebook copies are frozen as provenance inputs; the originals
in Documents/Trading/Codex are unchanged.

## The new comparison

Fixed Ridge_10 predicts native five-horizon targets using the established 20-by-2
EMA-change inputs and at most 252 mature examples. Its joint empirical distribution
uses the last 126 complete **earlier issued forecast-error vectors**, with all five
outcomes known by the current origin. Both research tracks are modeled separately.

`TS_null_empirical` uses the same calibration on the native null forecast. Its
residual-corrected distribution mean can be nonzero. It is a recent empirical
forecast comparator, not the zero-return point forecast. A separate zero/filter-
aware point control defines point-score skill.

The four trading candidates share the legacy NB9 support overlay. This keeps the
comparison opportunities aligned; the overlay is not an intrinsic ridge support
detector. The legacy analog also uses raw returns, volatility and volume coordinates,
whereas the ridge mean model uses smoothed changes. These are complete-procedure
comparisons with different libraries and features, not architecture-only tests.

All forecast scores use the same **654 daily origins per holding horizon**. The
original primary rule retains h=3, offset zero, giving 218 scheduled opportunities.
Unsupported and insufficient-edge cases remain explicit zero allocations.

## Costs, sizing and accounting

The inherited rule requires mean log return above the exact round-trip cost
threshold plus a five-log-bps buffer. Allocation is capped by 100%, a 10% annual
volatility target and a 1% tail budget. Trailing raw-return volatility is a risk
input; it does not enter the new ridge mean predictors. The mean-log threshold
is a heuristic, not a statistical significance test or expected-profit guarantee.

A budget wE buys wE/[P_entry*(1+c)] adjusted-price units. The exact isolated account
factor is `1 + w*((P_exit/P_entry)*(1-c)/(1+c) - 1)`. Sell before buying at a shared
roll close; never hold overlapping positions. Cash earns zero, shorting/leverage
are disabled, and position units can be fractional.

The assumed 5 bps per side lumps proportional trading costs. This does not model
actual quotes, closing-auction access, market impact, taxes, cash yield or guaranteed
fills. Drawdown is marked at daily closes and misses intraday losses. Adjusted-price
units are research bookkeeping, not executable historical share quotes.

## Primary hypothetical results

Common marked calendar: **2024-02-14 through 2026-09-25**, 656 observed closes
(655 daily returns), at 5 bps per side. These are **total** returns, not annual
returns. The window ends before the snapshot because the imported NB9 comparison
cohort requires all original five-session delayed outcomes to be known:

| Rule/distribution | Net total return | Maximum drawdown | Mean exposure | Completed trades |
|---|---:|---:|---:|---:|
| Legacy analog | 4.49% | -9.74% | 27.4% | 100 |
| Legacy recent | 0.99% | -15.01% | 54.6% | 194 |
| TS ridge empirical | 1.07% | -10.17% | 44.7% | 134 |
| TS null empirical | 3.87% | -7.35% | 52.4% | 166 |
| Buy-and-hold | 58.52% | -18.76% | 99.8% | 1 |
| Cash | 0.00% | 0.00% | 0.0% | 0 |

Buy-and-hold has much higher exposure. These raw return differences are not risk-
adjusted alpha. For ridge versus legacy recent, the mean paired daily net-return
difference is about **0.0011 bps**; the 20-session block range is approximately
[-2.55, 2.73] bps. All tested strategy-versus-recent ranges include zero at block
lengths 10/20/40. The samples were already examined and the ranges do not correct
for project-wide model searches.

Forecast and strategy rankings also differ. At the three-session holding horizon,
ridge's actual-return MSE skill against the zero point forecast is -0.2390; its
smoothed-change skill is -0.2668. A small positive historical strategy return does
not establish superior forecasts or an economically useful edge.

The cost plot freezes decisions and allocation fractions while changing costs.
Later quantities change with available equity. The horizon plot shows **all**
declared horizons/offsets; it does not choose a favorable schedule afterward.

## Research handoff and preserved reruns

`research_only_latest_handoff.csv` contains the latest ridge distributions at
holding horizons 1–4 for both tracks. The status is `research_only_no_order`; actual
future entry/exit prices and labels are absent. The origin is 2026-10-01, not today.
No live NB10 input is overwritten and no execution service is started.

The maintained copy also preserves the user's reexecuted TS09 notebook and its
refreshed manifest. Increasing Jupyter execution counts are accepted without
rewriting them. TS09's model cache remains usable after this version update when
its recorded model dependencies are unchanged; unrelated added modules and a
verified version-only initializer change do not force expensive refitting.

Inspect the saved `explicit_horizon_mapping.csv`, reproduction checks,
`delayed_forecast_scorecard.csv`, `strategy_metrics.csv`, decision schedules,
daily ledgers, cost/offset tables and `paired_strategy_uncertainty.csv` for details.
No independently established trading advantage follows from this historical
integration experiment.
