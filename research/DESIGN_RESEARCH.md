# Time-series forecasting library: research and proposed design

Research date: 2026-10-31. Status: Milestones 1–3 implemented in
`../forecasting_library/`: reusable causal filters, target/maturity contracts,
baseline scorecards, FFT/decomposition and nested classical-model experiments,
with executed TS00–TS06 notebooks. TS05 adds supervised tabular methods and TS06
adds compact MLP/LSTM/GRU/causal TCN seed ensembles. TS07 onward remain proposals.
No deployed forecasting advantage has been established.

## 1. Recommendation and project fit

Build a small, auditable Python package with a progressive Jupyter demonstration suite. Its distinguishing contribution should be the integration of causal smoothing, explicit forecast targets, chronological evaluation, and reproducible experiments. Use existing numerical and forecasting libraries behind adapters rather than reimplementing their optimizers and neural architectures.

Maintain two use cases: a broad educational laboratory covering the major model families, and a smaller set of candidates evaluated for stock-data research. A method can remain educational even when it fails the stock benchmarks. No family should be presumed profitable or superior before evaluation.

The current project already contains causal EMA and multiscale analysis in canonical NB2, mature-label analog forecasting in NB9, and a delayed-execution decision layer in NB10. Preserve those notebooks and their results. Give the new companion sequence independent TS numbering. Reuse provenance and temporal-validation conventions; importing NB9 analog models should preserve their label and entry-delay definitions.

## 2. Targets and information timing

Let x_t = log(P_t). A deployable smoother must generate s_t = S_theta(x_0,...,x_t), with configuration theta fixed under the declared calibration/update policy. This is an operational filtered series, not an observed true latent fair price.

Support these distinct tasks:

1. **Smoothed-series forecasting:** predict s_(t+h), or its change s_(t+h)-s_t, for h = 1,...,5 observed sessions. The realized target is the output actually generated at t+h, without data beyond t+h.
2. **Future-return forecasting using smooth inputs:** predict x_(t+h)-x_t using causal smoothed levels, slopes, residuals, and other available features. Raw future returns are evaluation labels; the model need not use raw price levels as predictors.
3. **Synthetic latent-signal forecasting:** predict a known latent process in simulations. Only here can denoising error against a known latent truth be measured directly.

Confirmed user decision: keep the first two tasks as equal research tracks. Give each track its own target definitions, appropriate baselines, model selection, and scorecards at horizons 1–5. Share data snapshots, causal smoothing implementations, and evaluation origins where feasible, but do not combine their losses into a single ranking or treat success in one as success in the other. The synthetic latent-signal task supports validation of both tracks.

Use observed exchange sessions, not five calendar days or automatically generated generic weekdays. Holidays change forecast dates. A five-step forecast covers the next five observations and is not always within seven calendar days.

Separate forecast time, target start, target end, data availability, and execution time. A forecast after close t cannot imply an executable fill at that close. Existing NB9/NB10 targets begin at next-close entry, x_(t+1+h)-x_(t+1); h=5 there ends at t+6. This is different from the five-step forecasting target above. Export an explicit mapping and choose a compatible trading horizon rather than relabeling either target.

If adjusted historical prices are used, record the adjustment convention and snapshot date. A present-day adjusted snapshot is not automatically point-in-time data; cross-ticker membership, corporate-action revisions, and survivorship require separate treatment.

## 3. A necessary baseline: filter mechanics

Smoothing changes the forecasting task and can create substantial predictability without predictive information about future market innovations.

For the recursive EMA

    s_t = alpha*x_t + (1-alpha)*s_(t-1),

assume the log-price process is a martingale with respect to the information available at t. Under squared-error loss,

    E[s_(t+h) | F_t] = (1-alpha)^h*s_t + [1-(1-alpha)^h]*x_t.

This follows by iterating the recursion and using E[x_(t+j)|F_t] = x_t. It is a mathematical null benchmark, not an empirical claim about stock dynamics. For alpha=0.2 and h=5, the forecast is exactly 0.32768*s_t + 0.67232*x_t. It generally beats simply holding s_t constant under this null because it accounts for the filter catching up to the last observed price.

For other fixed linear filters, propagate a no-change conditional mean price path through the saved filter state. For nonlinear/adaptive filters, distinguish that plug-in path baseline from a conditional expectation; simulation may be needed. Include drift and simple local-trend alternatives, with parameters estimated from eligible training data.

A smoother with a smaller target variance must not win merely because its RMSE is numerically smaller. Compare models against the same-target, filter-aware baseline. Compare different smoothers using matched synthetic denoising criteria and/or a common future-return target. A secondary ablation may restrict models and baselines to the smoothed history alone; label the information restriction explicitly.

## 4. Smoothing research programme

| Family | Proposed implementation | Main question / limitation |
|---|---|---|
| SMA, weighted MA | Trailing windows | Reference lag and noise suppression |
| EMA | Recursive fixed-alpha filter; explicit initialization | Stable reference; low-frequency delay is (1-alpha)/alpha sessions |
| DEMA, TEMA | If E is the EMA operator, DEMA=2E-E²; TEMA=3E-3E²+E³ | Reduce low-frequency delay through signed combinations; measure gain, overshoot and noise sensitivity |
| Adaptive MA | KAMA and explicitly defined adaptive-alpha variants | Does adaptation improve a predeclared tradeoff after tuning costs? |
| Local polynomial / SG | Fit trailing polynomial and evaluate at the newest observation | Endpoint behavior differs substantially from centered SG; low lag can accompany high variance |
| Butterworth | Low-order digital SOS design, forward-only filtering with saved state | Cutoff, phase/group delay, numerical stability, initialization |
| Bessel / minimum-phase FIR | Optional comparison experiments | Alternative phase or delay tradeoffs; match attenuation before comparing |
| Smoothing spline | Fit only a trailing/prefix window, retain its endpoint | Boundary sensitivity; archive the endpoint issued at each date |
| Local regression | Trailing local-linear or quadratic regression | Compare weighting, robustness, and derivative stability |
| Kalman/state-space | Filtered local level and local linear trend | Joint noise/state model; distinguish filtering from full-sample state smoothing |
| FFT low-pass/band-pass | Retrospective decomposition plus separately replayed prefix/rolling endpoint variants | Boundary effects, periodic continuation, ringing, spectral selection and revisions |
| Wavelets / SSA | Later educational decomposition modules | Causal reconstruction is a separate implementation problem |

SciPy exposes endpoint SG evaluation through `savgol_coeffs(..., pos=window_length-1, use='dot')`; dot these coefficients with the trailing window. Using the ordinary centered interior SG output for historical predictors would leak future observations. [1]

Butterworth design should use second-order sections and forward `sosfilt`, preserving state between updates. Forward-backward `sosfiltfilt` is useful for retrospective comparisons but its interior outputs use later observations. A prefix-only endpoint implementation can be causal, but it is a separate estimator with boundary effects, not a zero-phase live equivalent of the full-sample interior curve. [2–4]

A smoothing spline penalizes curvature while fitting observations. Interpolation alone need not smooth noise. SciPy's GCV criterion is not a substitute for chronological forecast selection under dependent financial data. Never fit a spline to the entire historical-plus-test sample before forecasting; never overwrite previously issued endpoints with the latest fitted historical curve. [5]

FFT is a computational representation, not itself a forecasting model. Whole-sample frequency truncation is retrospective. A trailing/prefix transform retaining only the newest endpoint can be causal, but endpoint behavior must be evaluated as such. Fourier extrapolation additionally assumes persistent estimated oscillations. Low-pass retains the low-frequency level/trend; band-pass removes DC, so a band-passed component is not a smoothed price level. Forecast and reconstruct trend plus components explicitly when appropriate.

Do not optimize lag alone: the identity filter has zero lag and no noise suppression. For fixed linear filters report frequency response, group delay over a stated passband, step overshoot, settling behavior, and white-measurement-noise gain sum(w_j²). This gain is a reference under an IID additive-noise model, not a universal financial-noise measure. Group delay is frequency-dependent and can be undefined at transfer-function zeros. [6]

Compare filters along a Pareto frontier: delay versus noise attenuation, endpoint error, overshoot, and downstream forecast skill. Match approximate bandwidth or attenuation rather than nominal window length. On synthetic signals include constant/ramp/piecewise trend, step changes, chirps, changing-period cycles, heavy-tailed noise, and noise-free controls. Retain unfiltered observations as a reference experiment.

## 5. Model catalogue and priority

| Tier | Methods | Role |
|---|---|---|
| Mandatory baselines | Persistence; zero return; random walk with drift; local linear extrapolation; filter-aware no-innovation path | Every benchmark |
| Classical core | AR, MA, ARMA, ARIMA; SARIMA/SARIMAX; SES, Holt/damped Holt, ETS, Theta | Transparent reference models; seasonality only where supported |
| State-space | Local level, local linear trend, dynamic regression | Candidate joint filtering/forecasting approach |
| Tabular ML | Ridge/elastic net; kNN; SVR; random forest; histogram/gradient boosting; optional LightGBM/XGBoost | Lagged supervised learning; nonlinear comparisons |
| Neural core | MLP, LSTM, GRU, causal TCN | Educational comparison and conditional research candidates |
| Modern extensions | N-BEATS, N-HiTS, DLinear/NLinear, PatchTST, TFT | Optional larger-model laboratory |
| Structural/decomposition | STL/MSTL + forecasting; dynamic harmonic regression; Prophet; SSA | Primarily demonstrations on suitable nonfinancial series |
| Multivariate/nonlinear | VAR/VECM where justified; existing analog forecaster; Gaussian process; regime-switching models | Selected demonstrations and project bridge |
| Uncertainty/volatility | Quantile models; residual simulation/bootstrap; ARCH/GARCH; adaptive interval calibration | Distribution and risk layer; GARCH forecasts conditional variance rather than supplying mean-price alpha |
| Foundation models | Optional Chronos and TimesFM adapters | Later comparisons with checkpoint/version and training-overlap caveats |

The catalogue is a staged scope, not a commitment to ship all methods in the first release. Seasonal methods and intermittent-demand methods belong on datasets with the corresponding structure; do not impose weekly stock-price seasonality merely because there are five trading days.

Compare direct horizon-specific models, recursive one-step models and multioutput five-step models. They have different error propagation and fitting costs. Recommend direct ridge/boosting and a five-output neural head as initial designs, retaining recursive ARIMA/ETS forecasts as natural comparisons. No universal winner is established.

For a single ticker, overlapping windows do not create independent examples. Start with compact LSTM/GRU networks, multiple fixed random seeds, chronological early stopping, and a strict compute budget. Consider pooled multi-ticker training only as a separate experiment with synchronized time splits and per-series scaling. Existing research demonstrates that simple linear models can be strong benchmarks on long-horizon datasets; it does not prove that any architecture wins for these five-session stock targets. [7]

## 6. Existing libraries: build adapters, not duplicate engines

| Backend | Proposed responsibility | Decision |
|---|---|---|
| NumPy/pandas/SciPy | Arrays, indexing, filters, spectra, interpolation | Core |
| statsmodels | Inspectable ARIMA/ETS/state-space estimation and diagnostics | Initial statistical backend |
| scikit-learn | Pipelines, tabular regressors, feature/estimator interfaces | Initial ML backend |
| StatsForecast | Efficient broader classical model catalogue | Optional batch/backend adapter |
| MLForecast | Lag transformations, ML forecasting, horizon-specific models | Optional scaling adapter |
| PyTorch + NeuralForecast | Neural architectures and probabilistic objectives | Optional neural extra |
| sktime | Broad forecasting interface, composition, reduction and evaluation | Alternative orchestration option; do not require it plus every other framework |
| Darts | Broad unified forecasting and probabilistic interface | Alternative for fastest broad demos; retain explicit project timing/provenance checks |

Current official documentation confirms broad classical coverage in StatsForecast, lagged-ML workflows in MLForecast, LSTM/GRU and modern networks in NeuralForecast, and forecasting interfaces in sktime/Darts. [8–12] My engineering recommendation is a small project-owned contract with statsmodels/scikit-learn first and optional Nixtla adapters. This recommendation is based on inspectability and staged dependency management, not a measured runtime comparison. Confirm compatibility in the intended environment before selecting exact pinned versions.

## 7. Package boundaries and contracts

Proposed structure:

    forecasting_library/
      pyproject.toml
      src/tslab/
        data/          # calendars, snapshots, validation, provenance
        smoothing/     # causal filters, replay, response diagnostics
        targets/       # target definitions, label availability
        features/      # lag/rolling features; training-only transforms
        models/        # small common contract and backend adapters
        evaluation/    # nested rolling origins, metrics, paired uncertainty
        uncertainty/   # quantiles, simulation, calibration
        experiments/   # budgets, configurations, registry, manifests
        reporting/     # plots, score tables, warnings, export
      notebooks/
      configs/
      tests/
      outputs/         # isolated run directories

Separate `fit`, `update`, `predict`, and serializable state. A smoother should expose warm-up, configuration, state reset, causal/revision behavior, and batch/stream equivalence. A forecaster should declare horizon support, required covariates, update/refit behavior, and quantile/sample capabilities. Missing optional backends should give informative errors without breaking the core.

Do not enforce causality through a boolean declaration alone: verify behavior by replay. Every forecast record should include run/model/filter IDs, ticker, origin, target-start/end, horizon, units, point prediction, quantiles where available, training cutoff, data-availability cutoff, and fit/update policy. Preserve failures and unavailable forecasts in a separate audit table.

Every run manifest should record data digests, source/adjustment and calendar conventions, configuration hash, code/package versions, seeds, target definition, selection history, runtime and hardware. Historical artifacts from distinct tickers or target definitions must remain isolated. Synthetic demos are explicit runs; market runs never silently fall back to synthetic data.

## 8. Evaluation protocol

1. Start with deterministic synthetic cases, stochastic negative controls, and known latent-signal simulations. Then use public nonfinancial series with trend/seasonality so the general-purpose library is tested beyond stocks.
2. Reuse the project's examined SPY snapshot only for development and regression checks. Existing notes identify that period as previously explored. It cannot become a pristine holdout by assigning a new filename.
3. Designate a new untouched interval or a prospective paper-forecast period before assessing final candidates. Exact assets, dates, minimum history and model budgets require a later experiment specification.
4. Use nested chronological rolling-origin evaluation. Select filter parameters, features and hyperparameters using inner training/validation only. Score the selected pipeline on the next outer interval. No shuffled splits. Rolling-origin evaluation supports separate multistep errors. [13–14]
5. Replay causal state continuously across an evaluation boundary using only observations as they arrive. Do not reset filters at each test origin unless that is the specified live procedure. Freeze or update parameters under an explicit schedule; never use future recalibration to revise earlier forecasts.
6. For supervised labels, eligibility is defined by label-end availability. At a fit issued after close t, a label ending at t can be used if processing timing permits; anything later cannot. A five-output training example must have all five outcomes matured. Derive split gaps from timestamps instead of inserting an arbitrary fixed gap. Legitimate trailing feature overlap with past training data is not itself leakage.
7. Use identical eligible evaluation origins for direct head-to-head scores, and separately report missingness, failures and operational availability. Do not silently drop difficult dates or retain only a model's successful fits.
8. Fit scalers/imputers and any feature-selection transforms on eligible training data. Known future calendar features are allowed; future prices, volume and volatility require their own forecasts. Quantify teacher-forcing versus inference behavior in neural models.
9. Report results by horizon, asset and predeclared time/regime slices, plus a predeclared aggregate. Keep all unsuccessful candidate results and the total search budget.
10. Use paired time-block resampling or an appropriate dependence-robust comparison for aligned loss differences. Five-step overlapping labels and filter memory create dependence. Choose block-length sensitivity, not IID error bars. Repeated model/filter selection invalidates an unqualified winner's p-value.

Point metrics: MAE, RMSE, training-scaled MASE/RMSSE where their denominator is nonzero, and same-target skill = 1 - loss(model)/loss(baseline). Do not use price-level R² or MAPE as the principal stock criterion. Direction metrics should concern a stated change/return target and include class prevalence and probability calibration where relevant.

Distribution metrics: pinball loss for quantiles; CRPS when full distributions/samples are available; empirical 50/80/95% interval coverage, width and interval scores. Pointwise horizon intervals are not simultaneous five-day path bands. Calibrate only on matured forecast residuals. Time dependence and distribution shifts prevent casually claiming ordinary exchangeable conformal coverage; adaptive approaches also require their specific assumptions and interpretation. [13,15]

Trading evaluation is a later gate: actual executable-price conventions, delayed fills, turnover, costs, drawdown, exposure and paired strategy uncertainty. Improvement on the smoothed target alone is insufficient. Retain original NB10 baseline results and preserve its distinctions between forecast evidence and decision-rule outcomes.

## 9. Verification and acceptance gates

Required correctness checks include:

- Prefix invariance with fixed configuration: appending or poisoning future observations cannot alter previously issued causal outputs or forecasts.
- Streaming/batch equivalence, state checkpoint/resume equivalence, and unchanged historical endpoint records.
- Constant/ramp behavior, exact EMA identities, filter stability, warm-up and boundary behavior.
- Correct session calendar, target indexing, return units, and next-close versus same-origin target separation.
- Poisoned immature labels, train-only transforms, and no future covariate access.
- Score formulas checked against simple definitions, prediction/quantile shapes and interval ordering.
- Snapshot hashes and ticker/target isolation; no automatic market-to-synthetic fallback.

Each notebook should run independently from a declared input snapshot, execute from a clean kernel, save its tables/figures/configuration, and receive visual inspection. A smoke-run validates implementation; repeated-seed simulation and chronological market experiments evaluate methodology. Neither alone establishes deployment readiness.

Candidate status should progress from `demo` to `research_candidate` to `paper_validated`, using a predeclared decision rule. An appropriate gate requires improvement over relevant baselines, uncertainty and sensitivity analysis, acceptable availability/runtime, and stability on independent assets/periods. Numerical thresholds should be specified before the final evaluation, not invented afterward.

## 10. Sequential notebook plan

| Notebook | Scope | Completion criterion |
|---|---|---|
| TS00 | Definitions, environment, data contracts and negative controls | Target/timing conventions demonstrated and checked |
| TS01 | EMA, DEMA/TEMA, adaptive MA; analytical responses | Filter-state and lag/noise diagnostics validated |
| TS02 | Endpoint SG/local regression, splines, Butterworth/Bessel, Kalman | Endpoint replay, causality and fair comparison demonstrated |
| TS03 | FFT/band-pass, decomposition and revision experiments | Retrospective and causal endpoint results explicitly distinguished |
| TS04 | Persistence, filter-aware nulls, AR/ARIMA, ETS/Theta/state-space | Common 1–5-step benchmark completed |
| TS05 | Ridge/SVR/trees/boosting; direct versus recursive | Leakage-safe ML pipeline and baseline-relative scores |
| TS06 | MLP/LSTM/GRU/TCN | Multiple seeds, frozen budgets, comparable targets |
| TS07 | N-BEATS/N-HiTS/linear neural/transformer demonstrations | Optional dependencies isolated; no assumed superiority |
| TS08 | Quantiles, simulation and interval calibration | Coverage/width/proper scores and limitations checked |
| TS09 | Cross-dataset and cross-asset benchmark | Nested selection and independent evaluation audit |
| TS10 | NB9 analog and NB10 trading integration | Explicit horizon mapping and provenance checks |
| TS11 | Optional foundation-model laboratory | Checkpoint provenance, overlap caveats and compute comparison |

First implementation milestone: TS00–TS02 plus reusable replay/filter modules and minimal null forecasts for both equal-priority tracks. Demonstrate target construction, label maturity, and separate scorecards for smoothed-series forecasts and future-return forecasts using smoothed inputs. Do not begin large neural comparisons until targets and endpoint behavior pass the correctness checks. Next milestone: classical models and ridge/boosting on the common evaluation harness, evaluated separately on each track. Expand coverage after that foundation is stable.

## 11. Research sources

These sources document mechanisms and available implementations; they do not establish stock-forecast performance for this proposed package.

1. [SciPy endpoint-capable Savitzky–Golay coefficients](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html).
2. [SciPy Butterworth design and SOS recommendation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html).
3. [SciPy forward SOS filtering and persistent filter state](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html).
4. [SciPy forward-backward SOS filtering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
5. [SciPy penalized smoothing splines and GCV](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html).
6. [SciPy group delay](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.group_delay.html).
7. [Zeng et al., Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504). Long-horizon benchmark evidence, not five-session equity validation.
8. [StatsForecast model API](https://nixtlaverse.nixtla.io/statsforecast/src/core/models.html).
9. [MLForecast official documentation](https://nixtlaverse.nixtla.io/mlforecast/index.html).
10. [NeuralForecast model catalogue](https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/overview.html).
11. [sktime forecasting workflow](https://www.sktime.net/docs/examples/forecasting/).
12. [Darts official documentation](https://unit8co.github.io/darts/).
13. [Hyndman and Athanasopoulos, forecasting tools: baselines, point/distribution scores and rolling-origin validation](https://otexts.com/fpp3/toolbox.html).
14. [scikit-learn lagged-feature forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html).
15. [Gibbs and Candes, Adaptive Conformal Inference Under Distribution Shift](https://arxiv.org/abs/2106.00170).
16. [statsmodels structural state-space models](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.html).
17. [Amazon Science Chronos repository](https://github.com/amazon-science/chronos-forecasting).
18. [Google Research TimesFM repository](https://github.com/google-research/timesfm).

Local project evidence reviewed: RESEARCH_ROADMAP.md; notebook_2_multiscale/README.md; notebook_9/README.md; notebook_10/README.md; filter/forecast definitions in research_tools/build_multiscale.py and research_tools/build_analogs.py. Synced reference files were not modified.


## Completion record — 2026-11-01

All twelve notebooks TS00–TS11 are delivered in package 0.8.0. TS11 uses real
pinned Chronos-Bolt tiny/mini checkpoints at contexts 64 and 256, with both forecast
tracks, matched controls, provenance/overlap caveats, calibrated intervals,
negative controls and CPU cost measurements. The primary historical stock panel
does not support promoting either checkpoint into trading. Larger/different
foundation-model families and prospective validation remain optional extensions.
