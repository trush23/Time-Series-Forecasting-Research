# Reading TS09 cross-dataset results

TS09 tests the same **procedure** on several datasets. Each dataset gets its own
fits and past-only model selection. It is not a model trained once on SPY and
applied without fitting to other assets. It is also not an untouched test: the
stock caches appeared elsewhere in this project, and climate series are public
teaching data.

## Universe, targets and units

The stock basket is SPY, MCHP, MRVL, TXG, GOOGL and NVDA. All use 1,443 common
observed sessions, 2021-01-04 through 2026-10-01, with 64 scored origins per horizon.
This is a convenience basket, with related technology names; it is not a random
sample of stocks. The snapshots use historical adjusted prices and contain no
point-in-time corporate-action record or executable fill prices.

Both research tracks remain separate:

- `smoothed`: future causal fast-EMA signal level.
- `returns`: actual cumulative future signal change. This is log return for stocks;
  it is a Celsius or ppm difference for the monthly climate datasets.

El Niño has 732 monthly Celsius observations and 28 scored origins per horizon.
CO2 has 444 contiguous monthly means from 1965 onward and 24 scored origins per
horizon. No log transformation is applied to physical temperatures or concentrations.
Five observations mean five stock sessions or five climate months. Climate period
ends are an observation-time convention; historical publication delays and revisions
are not reconstructed. These examples demonstrate library generality rather than
operational climate forecasting.

## Selection and fitting

The frozen representative menu contains AR(5) on smoothed differences, Ridge_10,
histogram boosting, GRU and the small NBEATS demonstration. Monthly datasets also
get a period-12 seasonal EMA-change candidate. Nulls remain eligible in every track.
Drift and slope diagnostics are reported but are not selection candidates.

Ridge, boosting and neural models receive the established 20-by-2 smoothed-change
window. AR(5) forecasts the smoothed series and uses the restricted historical
bridge for actual changes. These differences matter: the comparison concerns
complete procedures, not just architecture.

Selection uses earlier inner origins separately for every dataset and track, then
freezes the choice for an outer block. All five labels must mature before a training
example or inner outcome is eligible. Models refit at each forecast origin. Neural
families average three fixed seeds and preserve failures; no best seed is promoted.
The budget is at most 252 mature examples, so early climate fits can use fewer.

`TS09_PROTOCOL.json` records every declared input hash, schedule, candidate and
aggregation rule. It was frozen before the benchmark scores were computed. A
completed-model cache is reusable only when protocol, code, result and checkpoint
hashes match. Set `RECOMPUTE=True` in TS09 to refit the full model stage.

## Aggregation and uncertainty

Skill is calculated against the relevant null on the same origins and targets:
`1 - model loss / null loss`. The main null is filter-aware unchanged-price
propagation for smoothed levels and zero actual change for the other track.

The macro average gives each dataset equal weight **after** normalizing its loss
by its own null loss. Stock and nonfinancial aggregates are separate. Never pool
raw squared errors from log prices, Celsius and ppm. A favorable macro average can
hide individual failures; inspect the per-asset and chronological-block plots.

The stock uncertainty calculation samples the **same time blocks across all six
assets**. This preserves their contemporaneous dependence instead of treating them
as independent samples. It averages each asset's selected-minus-null absolute loss
after dividing by that asset's full-cohort null MAE. Those normalization denominators
stay fixed in the resampling. Negative differences favor selection.

Reported ranges depend on this historical moving-block design. They are not MSE-
skill confidence intervals, population-of-assets inference, future forecast bands
or corrections for project-wide exploration. Inspect all block lengths 1/2/4.

## Five-observation findings

The frozen selected procedure gives these MSE skills against its dataset-specific null:

| Dataset | Scored origins per horizon | Smoothed-level skill | Actual-change skill |
|---|---:|---:|---:|
| SPY | 64 | -0.0633 | -0.3585 |
| MCHP | 64 | -0.1089 | -0.1370 |
| MRVL | 64 | -0.1842 | -0.0674 |
| TXG | 64 | 0.0023 | 0.0419 |
| GOOGL | 64 | -0.1691 | -0.0161 |
| NVDA | 64 | 0.0004 | -0.0535 |
| El Niño | 28 | 0.8476 | 0.8030 |
| CO2 | 24 | 0.9844 | 0.9770 |

Equal-asset stock mean skills are **-0.0871** for smoothed levels and **-0.0984**
for actual returns. Only 2/6 and 1/6 asset scores are positive, respectively;
the two positive smoothed scores are very small. This result does not establish
that every possible stock model fails. It shows that this fixed menu and past-only
selector did not beat the declared nulls on average in this historical basket.

The physical monthly signals behave very differently. Their strong null-relative
scores do not establish state-of-the-art climate forecasting or transfer that
predictability to stocks. The seasonal benchmark is reported for context; the
selected procedure frequently chooses NBEATS_demo, with some boosting and seasonal
choices. The data have structured trend/seasonality and different horizons/units.

For blocks of four scheduled origins, the synchronized stock-panel normalized MAE
difference is +0.0256 with range [0.0033, 0.0529] for smoothed levels and +0.0254
with range [0.0005, 0.0593] for actual returns. Positive differences favor the null.
Ranges at block lengths one and two include zero for both tracks, so the historical
resampling interpretation is sensitive to block length. These are not confidence
bounds on the MSE-skill values or multiplicity-adjusted significance claims.

## What the independence audit establishes

Computational timing, paired cohorts, input identity and seed averaging can pass
without establishing independent research evidence. TS09 explicitly marks the
following as unestablished: a pristine asset/period holdout, point-in-time historical
vintages, completed prospective evaluation and a trading advantage.

A prospective specification is included for new forecasts issued no earlier than
2026-11-02. It is **not running or scheduled**. It requires forecast records and input
vintage hashes stored before outcomes, with no retrospective replacement. It does
not invent an acceptance threshold or promote a historical outer winner to a live
strategy. Deployment still requires a separately reviewed acceptance rule.

Open the saved `scores.csv`, `block_scores.csv`, `selections.csv`,
`macro_scorecard.csv`, `paired_uncertainty.csv`,
`synchronized_stock_panel_uncertainty.csv` and `independent_evaluation_audit.csv`
for the corresponding evidence.
