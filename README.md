# Time-Series Forecasting Library — Complete TS00–TS11 Program

The final notebook is [TS11 — Foundation-Model Laboratory](notebooks/TS11_Foundation_Model_Laboratory.ipynb).
**All twelve planned notebooks are complete.** Package version: **0.8.0**.
TS11 compares pinned Chronos-Bolt tiny/mini checkpoints at contexts 64/256 with
matched null and Ridge baselines across six stocks and two monthly climate series.
Both the future-smoothed and actual-return/change tracks remain equally represented.
Read [the foundation guide](FOUNDATION_MODEL_GUIDE.md) and
[Milestone 8 validation](VALIDATION_MILESTONE_8.md).

Open TS11 with the existing project kernel. Default audited replay requires no
Chronos installation or download; Ridge fits, scores, intervals and figures are
recomputed. `RECOMPUTE=True` reruns genuine local checkpoint inference with the
optional `requirements-foundation.txt` dependencies. Pinned weights are bundled.
In the primary stock experiment, both checkpoints underperform the filter-aware
null; no trading-model promotion is made. Pretraining data overlap is unknown.

TS00–TS10 are preserved, including the user's rerun of TS10. TS10's
[trading integration guide](TRADING_INTEGRATION_GUIDE.md) describes the unchanged
next-close horizon mapping and decision procedure.

The maintained copy is `Documents/Trading/Codex/Time_Series_Forecasting`, with one
`Time_Series_Forecasting.zip` alongside it. Subsequent deliveries update these same
paths. The publisher preserves user edits when the staged copy contains exactly
the same edited bytes, and stops on other conflicting changes. Extra user files
remain in the folder.

The complete suite uses `requirements.txt`, or
`python -m pip install -e '.[notebooks,test,classical,ml,neural]'` in a new environment.
The base requirements already existed in the validated environment. Optional fresh
TS11 inference additionally uses `requirements-foundation.txt`. `TS10_PROTOCOL.json`
records the frozen timing, candidate comparison, costs and risk settings.

## Notebook map

| Notebook | Main subject |
|---|---|
| [TS00](notebooks/TS00_Foundations_and_Targets.ipynb) | Definitions, timing, targets and negative controls |
| [TS01](notebooks/TS01_EMA_and_Adaptive_Filters.ipynb) | EMA, DEMA/TEMA and adaptive smoothing |
| [TS02](notebooks/TS02_Endpoint_and_State_Space_Filters.ipynb) | Endpoint polynomial/spline, causal filters and Kalman |
| [TS03](notebooks/TS03_FFT_and_Bandpass_Decomposition.ipynb) | FFT, band-pass and revision experiments |
| [TS04](notebooks/TS04_Classical_Forecasting_and_Selection.ipynb) | AR/ARIMA, ETS/Theta and state-space forecasting |
| [TS05](notebooks/TS05_Tabular_Machine_Learning.ipynb) | Tabular ML; direct versus recursive forecasts |
| [TS06](notebooks/TS06_Compact_Neural_Forecasting.ipynb) | MLP, LSTM, GRU and causal TCN |
| [TS07](notebooks/TS07_Advanced_Architecture_Demonstrations.ipynb) | Linear, N-BEATS/N-HiTS-style and transformer demos |
| [TS08](notebooks/TS08_Probabilistic_Forecasts_and_Calibration.ipynb) | Distributions, intervals and calibration |
| [TS09](notebooks/TS09_Cross_Dataset_and_Cross_Asset_Benchmark.ipynb) | Cross-asset/data benchmark and selection audits |
| [TS10](notebooks/TS10_NB9_NB10_Forecast_and_Trading_Integration.ipynb) | Forecast clock and trading integration |
| [TS11](notebooks/TS11_Foundation_Model_Laboratory.ipynb) | Pinned pretrained foundation-model laboratory |

## Cross-dataset benchmarking

[TS09 — Cross-Dataset and Cross-Asset Benchmark](notebooks/TS09_Cross_Dataset_and_Cross_Asset_Benchmark.ipynb)
compares a fixed menu across six stock histories and two monthly climate series.
Read [the benchmark guide](BENCHMARK_GUIDE.md) and
[Milestone 6 validation](VALIDATION_MILESTONE_6.md). `TS09_PROTOCOL.json` preserves
its experiment. The completed model stage is reused only when model dependencies,
protocol/results/checkpoints are unchanged; set `RECOMPUTE=True` for fresh fits.
Unrelated added modules and an audited version-only package initializer change
allow reuse without changing the cached model-stage provenance.

## Probabilistic forecasting

[TS08 — Probabilistic Forecasts and Calibration](notebooks/TS08_Probabilistic_Forecasts_and_Calibration.ipynb)
covers Gaussian/empirical residual distributions, quantiles, chronological rank
intervals and simultaneous bands. Read [the uncertainty guide](UNCERTAINTY_GUIDE.md)
and [Milestone 5 validation](VALIDATION_MILESTONE_5.md). Its 239-origin cohort differs
from earlier summaries; cross-notebook numerical comparisons require matched dates.

## Advanced architecture demonstrations

[TS07 — Advanced Architecture Demonstrations](notebooks/TS07_Advanced_Architecture_Demonstrations.ipynb)
provides five compact feature-to-target adapters: decomposition linear, last-centered
linear, N-BEATS-style residual stacks, N-HiTS-style multirate interpolation and a
patch transformer. Its architecture table identifies deviations from the published
models. Read [Milestone 4 validation](VALIDATION_MILESTONE_4.md).
The [performance guide](PERFORMANCE_GUIDE.md) explains TS05–TS07 point scores,
family ensembles, seeds, frozen procedures and historical uncertainty.

## Tabular and compact neural set

[TS05 — Tabular Machine Learning](notebooks/TS05_Tabular_Machine_Learning.ipynb)
and [TS06 — Compact Neural Forecasting](notebooks/TS06_Compact_Neural_Forecasting.ipynb)
cover ridge/Elastic Net/SVR/forest/boosting, direct versus recursive forecasting,
and compact MLP/LSTM/GRU/causal TCN models. Their accepted execution evidence is
preserved. Read [Milestone 3 validation](VALIDATION_MILESTONE_3.md).

## FFT and classical set

The next set is ready: [TS03 — FFT and Band-pass Decomposition](notebooks/TS03_FFT_and_Bandpass_Decomposition.ipynb)
and [TS04 — Classical Forecasting and Selection](notebooks/TS04_Classical_Forecasting_and_Selection.ipynb).
Both are executed, independently runnable notebooks with HTML reading copies.
The accepted TS00–TS02 notebooks and their saved outputs have been preserved.

TS03 adds causal FFT endpoints, low-pass/trend-plus-band decomposition, revision
audits, Fourier extrapolation and synthetic stress controls. TS04 adds ten
classical candidates: AR(1)/AR(5) on smoothed changes, MA(1), ARMA(1,1),
ARIMA(1,1,1), SES/ETS(A,N,N), damped Holt, nonseasonal Theta(2), local level
and local linear trend. Selection uses nested chronological validation with
separate choices for the two research tracks.

**New dependency:** statsmodels >=0.14.4. It is included in `requirements.txt`,
or install the local package with `python -m pip install -e '.[notebooks,test,classical]'`.
The validated environment already contains statsmodels 0.14.4; no new installation
was required for that delivered run. Milestone 2 used package version 0.2.0.

Read [Milestone 2 validation](VALIDATION_MILESTONE_2.md) for checks, results and limits.

## Start with the foundations

Start with [TS00 — Foundations and Targets](notebooks/TS00_Foundations_and_Targets.ipynb),
then [TS01 — EMA and Adaptive Filters](notebooks/TS01_EMA_and_Adaptive_Filters.ipynb),
and [TS02 — Endpoint and State-Space Filters](notebooks/TS02_Endpoint_and_State_Space_Filters.ipynb).
Each notebook has an executed HTML reading copy beside it and runs independently.

The first milestone implements causal smoothing, checkpoint/replay, data and label
contracts, and simple forecast baselines. Both research tracks have equal priority:

1. Future causal smoothed log levels at 1–5 observed sessions.
2. Actual future log returns, with a smoothed-slope candidate and explicit zero/raw-drift benchmarks.

Classical models and chronological selection are covered in TS04;
FFT/decomposition is covered in TS03; tabular ML and compact neural models are
covered in TS05–TS06. Adapted advanced architectures are covered in TS07;
probabilistic forecasts and chronological calibration are covered in TS08.
Cross-dataset and cross-asset evaluation is covered in TS09; NB9/NB10 integration
is covered in TS10; TS11 remains the optional foundation-model laboratory.

## Open and run

Keep this whole directory together. In Jupyter, select a Python environment with
NumPy, pandas, SciPy, Matplotlib and the notebook packages in `requirements.txt`.
The notebooks find `src/tslab` relative to their location; installation of the local
package is optional. Open TS00 and run all cells, then proceed to TS01 and TS02.
No network access or earlier notebook execution is required.

For an isolated new environment, install `requirements.txt`. For package use, install
the folder with `python -m pip install -e '.[notebooks,test,classical,ml,neural]'`.

The included execution helper runs one notebook per fresh Python process without
requiring a kernel socket:

```sh
python scripts/execute_notebook.py notebooks/TS00_Foundations_and_Targets.ipynb
python scripts/execute_notebook.py notebooks/TS01_EMA_and_Adaptive_Filters.ipynb
python scripts/execute_notebook.py notebooks/TS02_Endpoint_and_State_Space_Filters.ipynb
python scripts/execute_notebook.py notebooks/TS03_FFT_and_Bandpass_Decomposition.ipynb
python scripts/execute_notebook.py notebooks/TS04_Classical_Forecasting_and_Selection.ipynb
python scripts/execute_notebook.py notebooks/TS05_Tabular_Machine_Learning.ipynb
python scripts/execute_notebook.py notebooks/TS06_Compact_Neural_Forecasting.ipynb
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/verify_artifacts.py
```

The plugin setting keeps unrelated globally installed pytest plugins out of this
project's tests. The normal Jupyter Run All path remains supported. Exact validated
versions are recorded in `environment-validated.txt` and per-notebook manifests.

## What is implemented

`src/tslab/smoothing.py`: identity, trailing SMA/WMA, EMA/DEMA/TEMA, KAMA,
endpoint SG, recency-weighted local linear regression, fixed-lambda trailing
smoothing spline, forward SOS Butterworth/Bessel and filtered Kalman local trend.

`data.py`: explicit offline price snapshots with byte digests, strict time-series
validation, and a labeled synthetic generator. There is no implicit filling,
network download or market-to-synthetic fallback.

`targets.py`: target definitions, observed-session indexing, label availability,
all-horizon maturity, and a separately labeled next-close-entry bridge.

`evaluation.py`: chronological baseline predictions, origin-level records, separate
scorecards and paired moving-block MAE differences. Forecast routines receive only
past prefixes. Later labels are attached after forecasts. Scored origins match
within each filter/track/horizon; the latest immature forecasts remain in exports.

`diagnostics.py`: causal prefix/poison/state checks; fixed-LTI response, white-noise
gain, delay, step overshoot and settling; common-latent-target denoising scores.

`reporting.py`: figure conventions, configuration/source/software hashes and manifests.

`spectral.py` / `spectral_evaluation.py`: FFT decomposition, archived trailing
endpoints, checkpoints, past-only harmonic selection and separate target scorecards.
The harmonic extrapolator is unanchored: its fitted endpoint can differ from the
current issued smoother endpoint. Both that gap and the first predicted change are
exported in the audit; poor forecasts remain in the primary results.

`classical.py` / `prequential.py`: statistical model adapters, explicit actual-return
calibration, inner/outer chronological selection, parameter refits and visible
baseline fallbacks. Model fitting receives only the available prefix. Ground truth
is attached afterward. Each snapshot has its own in-memory fit cache.

`supervised.py`: smoothed-input windows, fully mature five-output training examples,
EMA anchor/residual targets, purged chronological stopping splits and recursive
EMA-state projection without teacher forcing.

`tabular.py` / `neural.py` / `learned_evaluation.py`: train-only scaling, direct and
recursive tabular prediction, compact neural networks, fixed seed ensembles,
nested selection and complete fit/failure audits.

`learning_reports.py` / `provenance3.py`: per-track/seed scorecards, paired uncertainty,
frozen prior-procedure comparisons and neural prediction-checkpoint hashes.

## Supervised experiment conventions

- Both tracks use only 20-observation sequences of fast/slow EMA increments
  (alpha 0.2/0.05). No raw-price feature, raw-return lag, centered smoother,
  future covariate or automatic imputation enters a learned model. Fixed EMA
  sequences are algebraically invertible, so this is a filtered representation,
  not an information-theoretic guarantee that high-frequency information is removed.
- Track A learns residuals above the exact EMA no-change anchor expressed from
  the current smoothed increment. Track B directly learns actual cumulative
  log returns. It no longer uses TS04's restricted return bridge.
- A training example is eligible only when all five labels have matured.
  At origin t the latest training feature origin is t-5. Use at most 252
  eligible examples for both direct and recursive models. The last observed
  close at t may enter a mature label; no later close can do so.
- Direct tabular models fit five outputs. Recursive ridge fits the one-step
  label on the same examples and generates both future EMA channels from its
  own predicted innovations. Actual future inputs are never fed back.
- Neural models train separately for each track and have five-output heads.
  MLP/LSTM/GRU/causal TCN use 713/833/641/1,481 parameters, respectively.
  The TCN uses strictly left-padded convolutions with dilations 1,2,4,8.
- Each neural fit uses a purged chronological 80/20 split for epoch choice:
  optimization labels end before the first stopping-validation feature origin.
  At the full 252-example budget, 196 examples optimize the network, 51 form
  the stopping tail and 5 are purged. Stopping scalers use optimization data only.
- Stop within 30 epochs, patience 5, AdamW learning rate .005, weight decay .001,
  full batches and gradient norm capped at 1. Reset the seed and refit a fresh
  network on all mature examples for exactly the selected epoch count, with
  final scalers fitted to that available pool.
- Three fixed seeds (1731/1732/1733) form an equal-weight ensemble per architecture.
  Seeds are never ranked for selection. A failed seed is retained with its
  baseline fallback; any incomplete ensemble uses the whole-ensemble fallback.
  Seed scatter is initialization variability, not a forecast/confidence interval.
- Both notebooks use outer positions 1000/1400/1800, three inner origins per
  block, and 40-session outer spacing. There are 30 scored outer origins per
  horizon plus the latest unscored forecast. This is a deliberately bounded
  teaching experiment; three inner origins provide noisy model-selection evidence.
- Selection minimizes equal-horizon relative MAE separately by track. This
  differs from displayed MSE skill and standardized neural stopping loss.
  The baseline remains eligible, and the chosen procedure stays frozen in each
  outer block. No best outer architecture, horizon or seed is promoted afterward.
- Frozen TS04/TS05 selected-procedure references live under
  `inputs/learning_references/`. They are checksum verified and restricted to
  exact common dates/targets. TS04 had different budgets and a different return
  design; comparisons concern complete procedures, not isolated architecture effects.

TS06 saves 24 final-origin inference checkpoints under its output `checkpoints/`
directory. Load trusted delivered checkpoints with `torch.load(..., weights_only=True)`
and `NeuralFit.restore`. They include scales, architecture, seed, feature configuration,
query sequence, anchor and saved predictions. They reproduce prediction, not the
exact continuation of AdamW training: optimizer state is not saved.

## Classical experiment conventions

- Track A forecasts the fixed EMA(alpha=0.2) target. Track B fits mature pairs of
  actual next-session returns against the current smoothed slope, then propagates
  those slopes from a model's native future smoothed path. It never labels a
  smoothed change as an actual return. This restricted two-stage bridge is explicit;
  all native families share the same horizon-one return estimate. Differences
  among those return candidates emerge at horizons two through five.
- Training windows contain the last 252 finite smoothed observations. Model
  normalization uses that window only. Bridge calibration uses at most 251 mature
  daily return pairs. Its last label ends at the current observed origin.
- Outer blocks begin at positions 1000, 1400 and 1800. Scheduled outer origins
  advance by 20 observed sessions; the final snapshot origin is added as an
  unscored forecast. These settings give 60 scored origins at each horizon.
- Each block uses six earlier inner origins, 25 sessions apart. Their maximum-
  horizon labels end before the first outer origin. Selection is separate by
  track, minimizing the equal-horizon average MAE relative to the corresponding
  inner baseline. Thus selection targets MAE; displayed MSE skill is a separate
  metric and need not improve even when inner MAE improves.
- A candidate needs at least 80% successful inner fits. The appropriate baseline
  is always eligible and can win. Model parameters are refitted at each origin;
  the chosen model remains frozen throughout that outer block.
- Failed or nonconverged fits use unchanged-path filter propagation for smoothed
  levels or zero for returns. Every failure is recorded, and its fallback loss
  remains in the scorecard. The selected procedure does not silently drop dates.
- The full candidate outer scorecards are exploratory comparisons. Only the
  predeclared inner-selected procedure represents the nested selection result.
  No best-looking outer family is promoted after observing its outer losses.
- Parameter estimates, fit warnings, chosen models, selection cutoffs and mature
  calibration counts are inspectable in the exports. No uncertainty bands or
  live trade decisions are supplied in this milestone.

## Scientific conventions

- `alpha` is an EMA smoothing rate, not a span. Recursive EMA cascades initialize
  at the first observed value. Default warm-up suppresses the first 59 outputs;
  windowed methods also require their entire trailing window. Warm-up is an explicit
  operational policy, not a proof that initial-condition error has vanished.
- IIR cutoff is in cycles per observation, with Nyquist 0.5. Bessel uses `norm='mag'`
  so its declared cutoff is a -3 dB amplitude cutoff. SOS state initializes at the
  first observation's steady state and persists across evaluation boundaries.
- KAMA uses a trailing efficiency ratio over `window` differences, spans 2/30 by
  default and squared smoothing rates. It has no fixed LTI response.
- Endpoint SG and local-linear fits evaluate at the newest observed time.
  The spline penalty uses integer session coordinates and fixed lambda; cached
  linear endpoint weights equal refitting that same spline to every trailing window.
  There is no GCV or future-dependent parameter calibration.
- Kalman uses state [level, slope], transition [[1,1],[0,1]], diagonal Q, observation
  variance R, initial mean [first value,0] and initial covariance diag(R,R).
  The first observation initializes the state; subsequent observations use filtering,
  not backward state smoothing. Q/R units are log-level/slope variance conventions.
- Smoothed-track forecasts are log levels; return forecasts are log differences.
  Multiplying a log difference or level error by 10,000 expresses log basis points.
  Exponentiating a predicted mean log level does not automatically give mean price.
- Filter-aware unchanged-path propagation is the exact conditional mean under a
  martingale for fixed linear filters. For adaptive KAMA it is a plug-in scenario,
  not generally an expectation. Kalman operational-output scenarios and latent-state
  forecasts are distinct targets.
- Track-specific skill is `1 - model loss / baseline loss`, with undefined skill
  retained as NaN if baseline loss is zero. Positive skill favors the candidate.
  Smoothed track baseline: filter_nochange. Return track baseline: zero.
- Different smoothers define different smoothed targets. Do not rank their raw
  forecast RMSEs. Common-latent denoising and common-return prediction are separate
  comparisons. Neither implies profitable executable trades.

## Data and outputs

`inputs/SPY_adjusted_demo.csv` freezes the existing local NB3 project export:
2,199 adjusted SPY sessions, 2018-01-02 through 2026-10-01. Its matching metadata
records source, adjustment, retrieval date and exact CSV SHA256. It is previously
examined historical data, not an untouched holdout or current market data. The
metadata's export-byte digest differs from upstream internal dataframe hashes.

Use `load_snapshot(explicit_csv_path, explicit_metadata_path)` for another input.
Supply the actual observed exchange-session dates. Duplicate, unsorted, invalid,
nonpositive prices or failed checksums raise errors. Future dates must be supplied
by a known exchange calendar; the library does not invent them from generic weekdays.
Label maturity is not feature validity: a later supervised learner must also require
that the origin's features have completed their warm-up.

Each notebook saves its own figures, CSV scorecards/forecast records, checks,
execution record and run manifest in `outputs/<notebook_name>/`. Re-running replaces
that notebook's files; copy/archive a run before changing settings. Input data and
the existing NB1–NB10 notebooks remain unchanged.

Read `VALIDATION.md` for the completed validation record and bounded findings.
`VALIDATION.md` records the original milestone 1 execution;
`VALIDATION_MILESTONE_2.md` records TS03–TS04, and `VALIDATION_MILESTONE_3.md`
records TS05–TS06. Source hashes are in the per-notebook manifests. Historical hashes for
accepted notebooks refer to their original run; the old initializer is retained
under `provenance/milestone_1_sources/` and `provenance/milestone_2_sources/`
for portable audit after the version updates.
The full research plan remains in `../forecasting_library_research/DESIGN_RESEARCH.md`.
