# TS11 — Foundation-model laboratory

TS11 completes the planned **twelve notebooks, TS00–TS11**. Package 0.8.0 adds
real, pinned Chronos-Bolt tiny and mini checkpoints, two context lengths, matched
null/Ridge comparisons, two target tracks, mature-residual intervals, paired block
ranges, a Gaussian random-walk control and a local computation comparison.
This is one foundation-model family at two sizes, not a complete foundation-model
survey. It does not modify the TS10 trading procedure.

## Open and run

Open `notebooks/TS11_Foundation_Model_Laboratory.ipynb` with the same Python kernel
as the earlier TS notebooks. The default `RECOMPUTE=False` replays the bundled,
digest-checked **real model inference**, and freshly computes Ridge fits, target
adapters, scores, intervals, checks and figures. No Chronos installation, network
access, previous notebook execution or hidden local cache is required in this mode.
The saved HTML is a reading copy.

To run the actual checkpoint inference again, install `requirements.txt` plus
`requirements-foundation.txt` in a separate Python environment, select that kernel,
and set `RECOMPUTE=True` in the setup cell. From an existing environment, use
`python -m pip install -r requirements-foundation.txt`, or install the project with
`python -m pip install -e '.[foundation,notebooks]'` alongside the full requirements.
The weights are already local. Keep this a separate environment if you do not want
the pinned torch/transformers dependencies to change another project's kernel.
The exact validated versions are in `environment-validated-milestone8.txt`.

The execution script can also run the notebook independently:

```bash
python scripts/execute_notebook.py notebooks/TS11_Foundation_Model_Laboratory.ipynb
python scripts/verify_foundation.py
```

An isolated environment that inherits packages but lacks Jupyter's HTML templates
may need its parent environment's `share/jupyter` directory in `JUPYTER_PATH`.
The standard existing project kernel already has those templates.

## Checkpoints and provenance

| Checkpoint | Actual parameter count | Revision | Weight bytes |
|---|---:|---|---:|
| amazon/chronos-bolt-tiny | 8,652,672 | a0e552de83495b5c28c14c71c374f3e33280b340 | 34,622,352 |
| amazon/chronos-bolt-mini | 21,236,096 | 251268337516a88e253628c43e1d26ec577b376b | 84,956,096 |

The [tiny model card](https://huggingface.co/amazon/chronos-bolt-tiny) and
[mini model card](https://huggingface.co/amazon/chronos-bolt-mini), configurations
and safetensors weights are bundled under `inputs/foundation_models`, with per-file
SHA256s and Apache-2.0 license information. The
[official implementation](https://github.com/amazon-science/chronos-forecasting)
provides the direct quantile API. No remote model code is executed.
Checkpoints are deliberately pinned rather than fetched from a moving `main` on
notebook execution. `prepare_foundation_inputs.py` is an explicit online preparation
script; rerunning it is a new input/protocol revision, not routine reproduction.

Exact pretraining end dates and membership of **each of our eight datasets** are
unknown. Neither publication date nor checkpoint commit date proves a training
cutoff. General vendor benchmark claims do not establish independence of this
project's panel. We use no held-out/prospective/clean-zero-shot designation here.

## Protocol and equal research tracks

`TS11_PROTOCOL.json` freezes both model sizes, contexts 64/256, primary context 256,
primary horizon 5, forty evaluation origins, fixed Ridge_10, residual window 32
and the absence of score-driven selection. The original pre-execution protocol is
retained in `provenance/TS11_protocol_initial.json`. After the initial run, only the
baseline's serialized name changed from `null` to `filter_null`, because pandas
otherwise interprets that literal CSV string as missing. Cohorts, parameters,
predictions, target definitions and metrics were unchanged by that correction.

The six stocks are SPY, MCHP, MRVL, TXG, GOOGL and NVDA, using the exact TS09
adjusted-price snapshot through 2026-10-01. ELNINO and CO2 are the same monthly
teaching datasets as TS09. Stocks use every fifth session; the two climate series
use every month. There are 40 calibration origins, 40 evaluation origins and one
latest unscored origin per dataset, at five future observed steps. Stock evaluation
origins run from **2025-12-12 through 2026-09-24**. Climate horizons are months,
not days. This cohort differs from TS09 and TS10.

Both tracks use smoothed inputs only. Foundation inputs are trailing centered EMA
levels, alpha .2. Ridge uses the established fast/slow EMA-difference features and
up to 252 mature examples, separately by track. These different features and the
foundation models' external pretraining mean this is a comparison of complete
procedures, not a controlled architecture-only comparison.

1. **Smoothed track:** predict future EMA levels with the native .5 quantile path.
2. **Actual-change track:** invert that deterministic path through the EMA recursion
   and subtract the actual current level reconstructed from the last two EMA values.
   For stocks the target is cumulative log return. For climate it is actual physical
   change. This plug-in path is not generally the conditional return mean or median.

EMA inversion has coefficients 5 and -4 at alpha .2 and can amplify errors. EMA is
invertible, so smoothed-only inputs do not by themselves remove raw information.
The null is filter-aware catch-up under a constant actual level, with zero actual
change. It is stronger than persistence of the current EMA.

Marginal quantiles cannot be subtracted into a joint return distribution. We retain
all native quantiles for inspection, use .5 unchanged for point forecasts, count
crossings, and avoid constructing native return bands from them. Both tracks and
all six procedures receive the same mature-residual interval mechanism: last 32
complete earlier five-error vectors, origin+5 no later than current origin,
empirical 10/50/90 percentiles. This produces residual-adjusted medians and 80%
intervals; these medians differ from the unadjusted point forecasts. Dependence,
overlapping monthly targets and small calibration samples prevent a coverage claim.

## Results and how to read them

Primary h=5, context=256; equal-weight mean of within-stock MAE skill versus the
matched null. Forty scored origins per stock, six stocks:

| Procedure | Smoothed-level MAE skill | Actual-return MAE skill |
|---|---:|---:|
| Fixed Ridge_10 | -6.33% | -2.49% |
| Chronos-Bolt tiny, 256 | -26.07% | -12.60% |
| Chronos-Bolt mini, 256 | -25.87% | -10.31% |

Positive skill would be better. Thus, neither checkpoint improves on the null in
this stock experiment. Both contexts and all horizons remain visible; we do not
select the most favorable alternative. Climate performance is different and is
reported separately, without pooling Celsius/ppm/log-price losses.

At h=5 the nominal 80% residual intervals cover approximately 75–77% of the stock
outcomes, depending on model and track. Coverage is read alongside width and
interval score. `interval_scores.csv` also retains lower/upper pinball losses.

`paired_block_ranges.csv` gives 2,000-draw h=5 moving-block ranges at 1/4/8 forecast
origins. Stocks share the resampled time blocks and receive equal weights. Baseline
normalizers are fixed per asset. Negative normalized MAE differences favor a model;
the primary model differences are positive. These are exploratory conditional
ranges without multiplicity adjustment. Climate assets align relative positions,
not common dates.

The random-walk control uses 128 independent generated histories per checkpoint
and computes exact conditional risk including future innovation variance. Return
conditional mean zero and EMA filter catch-up are the correct nulls in this control.
This is a negative control, not an estimate of stock-market dynamics.

## Computation and reproduction evidence

`foundation_compute.csv` records the original fresh CPU float32 inference, one torch
thread, batch 32, five forecast steps. The two tracks share each inference. Model
load time, parameter count and weight size are separate. Neural inference excludes
pretraining, installation, downloads, decoding/scoring and residual calibration.
Ridge timing includes both track fits; the batching and cost boundaries differ.
Neither peak RAM nor hardware-general speed superiority is claimed.

`foundation_quantiles.npz` and `inference_manifest.json` retain actual quantiles,
dependency versions, model timings, signal hashes, source-module hash, protocol
hash and cache hash. Default replay rejects changed provenance rather than silently
reusing it. `RECOMPUTE=True` regenerates the inference archive offline. Numerical
outputs can differ slightly across torch versions/hardware; identical hashes are
required only for the delivered cache, not promised for a new inference run.

The package contains the executed notebook, HTML, six inspected figures, seventeen
CSV outputs, audited weights/cache, unit checks and an independent artifact verifier.
Latest forecasts are wholly unscored. The supported conclusion is a completed,
reproducible foundation-model demonstration, not a validated trading advantage.
