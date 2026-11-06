# Milestone 3 — TS05 and TS06 validation

Validated 2026-10-31 using the existing Python 3.12 environment, scikit-learn
1.6.1 and CPU PyTorch 2.5.1. Exact versions are saved in
`environment-validated-milestone3.txt`. Package version: **0.3.0**.
No dependency installation, model download or market-data refresh was required.

## Execution and preservation

TS05 and TS06 ran independently in fresh Python/IPython processes, executing all
**18 code cells**, with **11 inspected figures and 32 CSV tables**. Their saved
notebooks include figures, tables and summaries derived from executed outputs.
HTML reading copies are included beside the notebooks.

TS00–TS04 and their saved outputs remain unchanged. Earlier milestone ZIPs remain
unchanged. Historical package initializer versions are retained under
`provenance/milestone_1_sources/` and `provenance/milestone_2_sources/` so their
execution manifests remain auditable after the version update.

## Correctness and saved-artifact checks

**98 tests passed**, including the prior 77 tests. Added coverage includes:

- Smoothed-only feature prefix invariance, poisoned-future immunity and full
  five-label maturity at the forecast origin.
- Exact equality of the smoothed-input EMA anchor with unchanged-path filter
  propagation, and the identity `one-step smoothed residual = alpha * actual return`.
- Purged chronological stopping splits and recursive EMA dynamics without
  realized future inputs or teacher forcing.
- Recovery of an independent planted linear map, train-only tabular scaling,
  and finite five-output predictions from every tabular specification.
- Repeat-seed neural prediction equality, prediction-checkpoint restoration,
  train-only stopping-stage scales and final-refit scale verification.
- Strict temporal causality within the left-padded TCN blocks.
- Prefix forecast equality under future poisoning for both ML and neural runners.
- Forced tabular fitting failure and a forced single-seed neural failure, retaining
  all member rows and using the declared whole-ensemble fallback.

The saved-artifact verifier independently checks actual return and smoothed-level
labels, source/configuration/output digests, model choices, fallback values,
selection cutoffs, all training-label cutoffs, scaler means, chronological stopping
boundaries and exact equality of successful ensemble forecasts to member averages.
All **24 final-origin neural checkpoints** reload with `weights_only=True` and
reproduce their saved query/residual/decoded predictions exactly in the validated
CPU environment. These are inference checkpoints, not optimizer-state resumptions.

The updated 0.3.0 wheel contains the current source modules. The complete delivery
includes frozen reference forecasts and the small historical source snapshots;
verification does not require an earlier checkout or executing previous notebooks.

## Feature and target definitions

The fixed input is a 20-by-2 window of fast/slow EMA first differences, with rates
0.2 and 0.05. Raw levels and actual-return lags are not learned predictors. Raw
observations are used to build causal smoothers and actual-return labels. EMA
histories remain mathematically invertible; smoothing is a representation choice,
not proof that high-frequency information is irretrievably removed.

Track A learns corrections to the exact EMA no-change anchor, then adds the anchor
and current smoothed level back. Track B learns actual cumulative log returns
directly; it no longer uses TS04's restricted return bridge. Every example has all
five labels matured, so the final training feature origin is at most t-5 for a
forecast issued after observation t.

Use at most 252 mature training examples at each origin. Direct models predict five
outputs; recursive ridge fits its one-step output on those same examples and
generates future fast/slow EMA increments using its own predictions. All train/target
scaling is fitted on the eligible rows; histogram boosting's random early-stopping
split is disabled. Tabular numerical thread pools are bounded to one thread.

## Neural training and stopping

MLP, LSTM, GRU and causal TCN contain 713, 833, 641 and 1,481 trainable parameters,
respectively. Each track is fitted separately. The TCN uses strictly left-padded
convolutions with dilations 1,2,4,8; its 31-position receptive field covers the
20-position input with zero left padding.

Each seed fit first selects its epoch count using the latest chronological 20%
of available examples. Optimization labels must end before the first validation
feature origin. At the full budget this gives 196 optimization rows, 51 validation
rows and 5 purged rows. Overlapping trailing input histories are permitted; future
labels are not. Stopping-stage scalers use optimization rows only.

Training uses full-batch AdamW, learning rate 0.005, weight decay 0.001, gradient
norm capped at 1, at most 30 epochs and patience 5. A fresh network with the same
seed then refits all mature examples for exactly the chosen epoch count, using
scalers fitted to that available pool. Standardized MSE controls stopping; it is
not the same objective as the MAE-based model selector or reported MSE skill.

Seeds 1731, 1732 and 1733 form an equal-weight ensemble for each architecture.
Seeds are not ranked for selection. The stock experiment contains **960 seed fits**
(each with stopping and fresh refitting) and **320 complete ensemble forecasts by
origin/model/track**, including inner and latest unscored origins. All succeeded
in this run. TS05 contains 560 stock origin/model/track fits, also all successful.
Synthetic controls and coefficient diagnostics are additional separate computations.

Chosen refit epochs range from 1 to 30. Median choices are 8 for MLP, 15 for LSTM,
14 for GRU and 6 for TCN across tracks, origins and seeds. A low stopping loss can
also reflect lower volatility in the validation interval; it alone does not
establish predictive skill.

## Chronological selection and comparison

Outer blocks begin at observed positions 1000, 1400 and 1800. Each block uses three
earlier inner origins, 40 sessions apart, whose five-session labels mature before
the block begins. Model selection minimizes equal-horizon relative MAE separately
by track, with the relevant baseline always eligible. Outer origins advance by
40 sessions: **30 scored origins per horizon**, plus the latest unscored origin.
Three inner origins provide noisy selection evidence; this is a bounded teaching
experiment, not an exhaustive or deployment-grade tuning study.

Frozen TS04/TS05 selected procedures are compared on exactly common dates and
targets. Their original choices remain fixed. TS04 used different budgets and a
return bridge; TS05/TS06 learn actual-return targets directly. These are complete-
procedure comparisons, not controlled isolation of architecture effects.

## Bounded numerical findings

All stock results use the previously examined adjusted SPY snapshot ending
2026-10-01. There is no newly untouched test period.

| Selected procedure | Track | Five-session RMSE (log bps) | MSE skill vs baseline |
|---|---|---:|---:|
| TS05 tabular | Smoothed level | 103.27 | -0.0679 |
| TS05 tabular | Actual return | 203.83 | -0.0606 |
| TS06 neural ensemble | Smoothed level | 91.29 | 0.1655 |
| TS06 neural ensemble | Actual return | 195.30 | 0.0262 |

The smoothed baseline is unchanged-price propagation through the filter; the return
baseline is zero. The metric is `1 - model MSE / baseline MSE`, on identical dates
and targets. Positive values here are historical loss reductions, not profits.

TS06's selector chooses MLP in the first block for both tracks, then GRU for the
smoothed track and zero return for the return track in the remaining two blocks.
No favorable outer architecture or individual seed is substituted afterward.

For selected-minus-baseline **MAE** at horizon five, TS06's smoothed-track moving-block
95% percentile ranges are below zero for the three tested block lengths. The return-
track ranges include zero. For example, with blocks of four scheduled origins:

- Smoothed: mean difference -8.23 log bps; range approximately [-15.14, -0.75].
- Returns: mean difference -0.24 log bps; range approximately [-2.06, 8.57].

These are conditional historical MAE comparisons, not confidence bounds on the MSE
skill numbers, simultaneous forecast bands or multiplicity-adjusted claims. They
depend on the specified resampling assumptions and do not account for prior project
exploration or arbitrary nonstationarity. All tested TS05 paired MAE ranges include zero.

Seed scatter measures initialization variability on the same data. It is not an
independent sample of markets, a forecast distribution or a statistical confidence
interval. A single asset with 30 outer origins cannot establish a general advantage.

## Visual inspection and reproduction

All eleven new figures were inspected as rendered images; the calibration heatmap
colorbar was repaired and rechecked at full size. The neural scorecards, learning
curves, seed scatter, matched-date comparisons and paired intervals were reviewed.
Summary values were reconciled with saved tables.

HTML copies contain the executed figures/tables and descriptive image alt text.
Browser layout and MathJax rendering remain unverified because the browser surface
previously rejected local `file:` previews; no workaround was attempted. Actual
notebook figure rendering was inspected directly.

From the library directory with the full requirements installed:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS05_Tabular_Machine_Learning.ipynb
python scripts/execute_notebook.py notebooks/TS06_Compact_Neural_Forecasting.ipynb
python scripts/verify_artifacts.py
```

`build_learning_notebooks.py --only ts05` or `--only ts06` regenerates only that
notebook as unexecuted cells. It never rewrites TS00–TS04. Delivered frozen references
are already included; their preparation script is not needed for ordinary reruns.
Different hardware or library versions may change seeded training results.

## Remaining scope

TS07 can add larger neural architectures. Probabilistic forecasts, interval
calibration, broader assets, a locked prospective period, execution and transaction
cost analysis remain separate stages. No live forecast service or trading rule is
established by these notebooks.
