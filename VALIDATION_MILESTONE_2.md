# Milestone 2 — TS03 and TS04 validation

Validated 2026-10-31 using the existing Python 3.12 environment and statsmodels
0.14.4. Exact versions are in `environment-validated-milestone2.txt`.
Package version: 0.2.0. No environment installation or data download was required.

## Delivery and preservation

- TS03 and TS04 execute independently, with **16 code cells, 10 figures and
  23 CSV tables** in total. All cells ran in order in separate fresh Python/IPython
  processes; no cell computation was skipped or replaced.
- Their executed `.ipynb` files and generated HTML reading copies are saved together.
- The accepted TS00–TS02 notebooks and saved outputs were not regenerated or modified.
  The original milestone 1 ZIP remains unchanged. The new ZIP contains the full
  five-notebook library and updated source/package metadata.
- The package initializer's version changed from 0.1.0 to 0.2.0. Its prior bytes
  are retained under `provenance/milestone_1_sources/` so the original execution
  manifests can be audited portably. Other original computational modules are unchanged.

## Correctness checks

**77 tests passed**, including the 54 milestone 1 checks. Added tests cover:

- Exact FFT component reconstruction and recovery of a known in-band sinusoid;
  band-pass DC exclusion and constant/ramp controls.
- Causal FFT prefix invariance, future poisoning, state serialization/resume and
  nonmutating future-path projection.
- Exact harmonic forecasts for a known periodic signal and a linear trend.
- Classical-model constant paths; AR forecasts against an independent AutoReg fit;
  recovery of a known AR coefficient within a declared tolerance.
- Recovery of a known synthetic return-calibration relationship, with its final
  training label ending at the current origin.
- Identical candidate predictions under future-data poisoning, including spectral
  stock forecasts and native classical forecasts.
- Forced fitting failure with exact declared fallback predictions; chronological
  selection cutoffs and unchanged earlier-block selections under later-data poisoning.
- Rejection of immature inner labels and malformed spectral configurations.

The saved-artifact verifier checks all five notebooks for structure, sequential
execution counts, error-free outputs, embedded/exported figure counts, alt text,
configuration/source/output hashes and exact actual-return labels on the observed
session index. For TS04 it additionally verifies inner-label availability, bridge
label maturity, fallback values, frozen per-block model choices and exact equality
of selected predictions with the corresponding candidate or baseline predictions.

## Experiment definitions

TS03 uses fixed 128-session low-pass and trend-plus-band endpoint estimators;
Fourier extrapolation selects two frequency bins from the last 128 **smoothed
endpoints** using only their past amplitudes, then jointly refits trend and harmonic
coefficients. It is explicitly unanchored. Its current fitted-endpoint discrepancy
and first predicted change are exported, rather than hidden through future correction.

TS04 has ten candidate specifications and three outer blocks. At each block start,
six earlier chronological validation origins determine one chosen model per track.
Every five-session inner outcome has matured before that outer block begins.
The criterion is equal-horizon relative MAE, with an 80% minimum successful-inner-fit
rate. The corresponding baseline is always eligible. This criterion differs from
the MSE skill also shown in the figures.

Native smoothed-series predictions and actual-return predictions are separate.
The latter use an explicit OLS bridge from current smoothed slope to next actual
return, trained on mature labels. Native forecast paths supply future lagged
slopes. This restricted two-stage design gives every native family the same
horizon-one return estimate; family differences appear from horizon two onward.
It does not equate smoothed changes with actual returns.

All stock examples use the same 2,199-row adjusted SPY snapshot ending 2026-10-01.
It has already been examined in the project. No pristine holdout is claimed.

## Bounded numerical findings

TS03's synthetic historical FFT curve revises by up to **411.21 log bps** over
the old prefix after appending 32 observations. Archived causal endpoints have
**zero historical revision**. This measures revision under the stated synthetic
signal and hard frequency mask, not a market revision statistic.

At horizon five, the unanchored harmonic candidate has negative MSE skill in both
stock tracks, for both filter variants. Large smoothed-target errors are retained;
they expose endpoint mismatch and unstable extrapolation in this declared model.
This does not establish that all FFT-based approaches fail.

For TS04, the **selected procedure**, scored on 60 outer origins per horizon, has:

| Track | Five-session RMSE (log bps) | MSE skill vs baseline | Baseline |
|---|---:|---:|---|
| Smoothed log level | 129.49 | -0.1894 | Filter-aware unchanged-price path |
| Actual log return | 266.80 | -0.0273 | Zero return |

Thus neither selected procedure improves five-session MSE in this demonstration.
The return selector chooses the zero baseline in two of the three outer blocks.
Do not replace the selected procedure's result with the best outer family after
reading the scorecard. The values are losses relative to the same target and dates,
not profits or loss limits.

There are 790 unique origin/model native fits, including inner, outer and latest
unscored origins. **Seven do not converge**, yielding 14 track-level fallback rows
in the fit audit. Their outcomes remain in operational loss calculations. These are
expected, recorded fitting failures, not notebook execution failures.

Paired moving-block bootstrap 95% ranges for selected-minus-baseline MAE include
zero at all three tested block lengths for both tracks. The ranges are conditional
on the fixed procedure, previously examined period and block-resampling assumptions;
they do not adjust for earlier research choices or establish equality of methods.

## Presentation inspection

All ten new figures were inspected directly as rendered images, with full-size
inspection of the FFT revision and classical scorecard figures. Axes, units,
legends, synthetic controls, failure-rate bars and uncertainty intervals were
reviewed. Values and observed-result summaries were checked against saved tables.

HTML copies were generated and their embedded outputs checked programmatically.
Browser layout and MathJax rendering remain unverified, as the browser surface
previously rejected local `file:` previews; no workaround was attempted. The actual
notebook figure rendering was inspected directly.

## Reproduction

Using a Python environment with `requirements.txt`, from this library directory:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS03_FFT_and_Bandpass_Decomposition.ipynb
python scripts/execute_notebook.py notebooks/TS04_Classical_Forecasting_and_Selection.ipynb
python scripts/verify_artifacts.py
```

`scripts/build_next_notebooks.py` regenerates only TS03–TS04 as unexecuted notebooks;
run the execution helper afterward. It does not rewrite TS00–TS02.

## Remaining scope

TS05 adds tabular ML and TS06 adds compact neural models. Seasonal/exogenous
classical variants need separately specified datasets and candidate grids.
Probabilistic forecasts, simultaneous path bands, fresh market data, live execution
and a prospective independent evaluation remain outside this milestone.
