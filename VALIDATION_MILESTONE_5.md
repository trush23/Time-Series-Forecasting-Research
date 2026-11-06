# Milestone 5 — TS08 validation

Validated **2026-11-01** using the existing Python 3.12 environment, NumPy 1.26.4,
pandas 2.2.3, SciPy 1.13.1 and scikit-learn 1.6.1. Exact versions are recorded in
`environment-validated-milestone5.txt`. Package version: **0.5.0**.
TS08 requires no new dependency, market-data download or pretrained weights.

## Delivered scope and preservation

TS08 implements Gaussian and empirical residual distributions, seven quantiles,
rolling absolute-error and volatility-scaled rank intervals, a separate maximum-
normalized-error five-horizon band, proper marginal scoring and residual-vector
simulation. Both the future smoothed-level and actual-return tracks have separate
fits, calibration, scorecards and figures. Fixed Ridge_10 is a teaching point model,
not a retrospective winner promoted from earlier outer scores; each track also has
its own null point forecast and independently calibrated null distribution.

All **229 accepted notebook/output files from TS00–TS07** match the Milestone 4
archive byte-for-byte. Earlier delivery archives remain unchanged. The earlier
0.4.0 initializer is preserved under `provenance/milestone_4_sources/`; earlier
historical source snapshots remain bundled. No file under `sources/` was changed.

## Execution and correctness

TS08 executes independently, top-to-bottom in a fresh Python/IPython process:
**8 code cells, 7 inspected figures and 21 CSV tables**, with no error or warning
streams. The saved notebook contains its outputs and a data-derived summary;
an HTML reading copy is provided. Execution took about 27 seconds in this environment.

**120 tests pass**, including the prior 107 and 13 new cases covering:

- Pinball, interval score and WIS against hand-computable examples; Gaussian CRPS
  against its analytic standard-normal value; empirical CRPS against the independent
  pairwise absolute-distance formula.
- Corrected order-statistic rank and infinite radius when the sample cannot support
  the requested finite rank; interval nesting, ordered empirical quantiles and
  maximum-score bands containing their marginal counterparts.
- Exact eligible calibration origins, use of earlier issued forecast errors,
  delayed label maturity and exclusion of poisoned immature errors.
- Equality of issued point predictions, scale proxies and calibrated bounds after
  altering only future observations.
- Reproducible independent simulations, near-nominal IID behavior, deliberate
  under-coverage after a scale jump, and oracle-control scaling invariance.
- Invalid calibration/training budgets, probability ordering and simulation budgets.

The saved-artifact verifier checks all nine notebooks, including configuration,
input/output/source hashes, sequential outputs, figures and label availability.
For TS08 its independent audit reconstructs:

- Every ledger actual-return and smoothed-level target, each null forecast, every
  successful training pool/scaler mean and every causal scale proxy.
- Every calibration slice from the issued forecast ledger, retaining only complete
  vectors ending by the current origin; every interval bound and Gaussian/empirical
  quantile from those errors.
- Coverage, width, interval-score formulas, WIS, CRPS, distribution-score summaries,
  pinball losses and simultaneous-event coverage.
- Every saved residual-bootstrap draw as an intact native five-error vector.

The 0.5.0 wheel is built offline and checked against the current source modules.
The cumulative ZIP is extracted into a fresh temporary directory, all checksums
are checked, and the saved-artifact verifier is run there without the original
checkout or prior archives. The wheel imports directly from that extracted copy.

## Forecast and calibration timing

The same fixed 20-by-2 history of EMA first differences is used, with rates .2/.05.
Fit a direct Ridge(alpha=10) separately for each track on at most 252 examples whose
entire five-horizon label vector has matured. Training scalers use only that pool.
Track A predicts the residual around the exact unchanged-price EMA catch-up anchor;
track B predicts actual cumulative log returns. Raw levels and actual-return lags
are not fitted predictors. Raw observations still supply target labels.

Daily issuance begins at observed position 650 and ends at 2198: **1,549 origins**,
**3,098 successful ridge fits** and **30,980 ledger rows** including both nulls.
Operational fit failures would be recorded and use the declared null fallback;
no such failure occurred. Calibration uses the latest **126** complete daily
forecast-error vectors with `calibration_origin + 5 <= forecast_origin`, even at
horizon one. Its residual is actual minus the forecast issued earlier, not an
in-sample residual calculated with the present model.

Outer scoring uses positions 1000,1005,...,2190: **239 complete origins per horizon**.
The latest position 2198 is appended as wholly unscored; partially matured tail
origins are omitted from this common evaluation cohort. The daily ledger retains
all available/immature labels. Saved interval and quantile tables contain 72,000
and 67,200 rows, respectively, including the latest unscored query. There are
960 calibration audits. Calibration is recomputed at 252 errors on exactly the
same point forecasts and outer cohort for sensitivity; no outer-score choice is made.

## Distribution methods and limitations

Gaussian residual forecasts use the historical error mean and sample standard
deviation (floored at 1e-8). Empirical forecasts use the exact 126-atom residual CDF
and inverse-CDF quantiles. They can shift the forecast median; symmetric rank
methods retain the point forecast as center. Rank radius is the
`ceil((n+1)*coverage)`-th absolute-error order statistic, without interpolation or
clipping an unsupported rank to a finite value.

The scale proxy is RMS fast-EMA changes over 60 observations, floored at 1e-6,
multiplied by sqrt(h) for returns or sqrt(sum((1-.8^k)^2)) for smoothed levels.
It is a causal proxy, not a proved conditional standard deviation. Past errors are
normalized by their own origin's scale; the query uses its current scale. The joint
method ranks each complete error vector's maximum normalized error and reports
coverage of all five future targets together. Ordinary marginal interval scores
are not labeled proper joint scores; those joint-method score entries are missing.

Intervals have levels 50/80/95%; quantiles are .025/.1/.25/.5/.75/.9/.975.
Pinball uses the convention without an additional factor of two. WIS uses median
weight .5 and interval weights alpha/2, divided by K+.5. CRPS is the analytic
Gaussian score or exact finite-empirical-CDF score, not an unbiased ensemble-score
variant. Rank methods have no full CDF, so CRPS is missing rather than zero.
Scores and widths are log units; displayed log basis points multiply by 10,000.

All calibration residuals are prequential, but they overlap across origins and
come from refitted models. Training and calibration date ranges can overlap.
No ordinary exchangeable split-conformal guarantee is asserted. TS08 does not
implement adaptive conformal feedback or claim the guarantees of cited adaptive
algorithms. A 2.5% empirical tail has only about 3.15 observations before accounting
for dependence. Complete residual-vector resampling preserves the finite mixture's
cross-horizon dependence; it is not a new simulated stock-price process or a jointly
calibrated distribution across the two research tracks.

## Bounded findings

On the previously examined adjusted SPY snapshot ending 2026-10-01, Ridge's five-
session volatility-scaled intervals give:

| Track | Nominal marginal coverage | Observed coverage | Mean width, log bps | WIS skill vs same-method null |
|---|---:|---:|---:|---:|
| Smoothed level | 80% | 80.8% | 345.0 | -0.1464 |
| Actual return | 80% | 77.8% | 691.8 | -0.1188 |

Coverage near nominal does not establish useful forecasts: the corresponding null
distributions have lower WIS. Gaussian and empirical CRPS skill is also negative
for Ridge at horizon five. The fixed point Ridge has negative five-session MSE
skill (-0.3027 smoothed, -0.2383 actual returns) on this **different 239-origin cohort**;
these values must not be compared directly with the earlier 30-origin summaries.

Joint coverage of the five marginal scaled 80% intervals is 67.8%/56.1% for
smoothed/returns. The separately calibrated joint-max bands give 80.8%/78.7%, with
wider rectangles. Doubling calibration history to 252 gives 80.8% scaled marginal
coverage in both tracks and mean widths of 328.0/654.2 log bps. This is sensitivity
evidence, not a preselected superior window or proof of stability.

Independent simulation uses 200 replicates per track, 126 calibration vectors and
500 test vectors. At nominal 95%, absolute-rank marginal coverage is approximately
95.3% under IID Gaussian innovations and approximately 49.8%/49.7% after an abrupt
threefold increase in test volatility. Oracle scaling is explicitly labeled a
control, not the stock-data proxy. Averages over independent calibration replicates
are not evidence that one realized stock calibration window has guaranteed coverage.

The snapshot has been examined previously; there is no untouched or prospective
test here. Shared training/calibration windows, overlapping daily errors, one asset
and nonstationarity limit interpretation. No trading rule, cost model or market
advantage is established. Latest saved distributions refer to 2026-10-01, not today.

## Visual inspection and reproduction

All seven figures were inspected directly, and the width/coverage annotations were
simplified for readability. The distribution plot explicitly shows null-relative
WIS skill so near-nominal coverage is not mistaken for predictive superiority.
The seven views cover IID/shift controls, marginal calibration, width/coverage,
proper scores, joint coverage, intact-vector samples and local coverage variation.

HTML embedded figures, counts and alt text are checked programmatically. Browser
layout and MathJax rendering remain unverified because local file previews were
previously rejected by the browser surface. Actual exported figures were inspected.

From the library directory with its requirements installed:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS08_Probabilistic_Forecasts_and_Calibration.ipynb
python scripts/verify_artifacts.py
```

`build_probability_notebook.py` generates TS08 only as unexecuted cells. No earlier
notebook execution or network access is needed. `UNCERTAINTY_GUIDE.md` explains the
results. TS09 cross-dataset/cross-asset evaluation is the next planned stage;
conditional quantile learners, GARCH, adaptive conformal feedback and distributional
neural training remain optional extensions rather than implemented TS08 features.
