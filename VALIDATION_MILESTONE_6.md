# TS09 — Cross-dataset and cross-asset validation

Validated **2026-11-01**, package **0.6.0**, using the existing Python 3.12,
statsmodels 0.14.4, scikit-learn 1.6.1 and CPU PyTorch 2.5.1 environment.
`environment-validated-milestone6.txt` records exact versions. No new installation,
market-data request or public-data download was needed.

## Delivery and preservation

The maintained folder is `/Users/trush/Documents/Trading/Codex/Time_Series_Forecasting`,
with the same-name ZIP alongside it. Subsequent builds update those same paths.
The publisher verifies all checksums, preserves extra user files and stops on edits
to previously managed files. Historical source snapshots used by accepted notebooks
are included, so replacing the stable ZIP does not erase execution provenance.

All accepted TS00–TS08 notebooks and output files were checked byte-for-byte against
the previous delivery and remain unchanged. Old milestone archives are unchanged.
No synced file under `sources/` was modified.

## Scope and data audit

`TS09_PROTOCOL.json` was frozen before scoring, SHA256:
`dfc11ef022e13e0ce560ea7a62a8477f3a7b2bef6ee1b9c15531826a890a3432`.
It records the exact input hashes, universe, schedules, candidates, seeds, features,
aggregation and uncertainty rules. Notebook assertions verify it is unchanged.

Six stocks—SPY, MCHP, MRVL, TXG, GOOGL and NVDA—share 1,443 actual observed sessions,
2021-01-04 through 2026-10-01. All cached source byte hashes were verified before
transformation. The first five caches were adjusted explicitly with provider
Adj Close / Close; producer code evidence is bundled. NVDA's source declares
auto_adjust=True. These are revised historical adjusted values, not executable
quotes or point-in-time corporate-action records. All six stock calendars match;
no implicit intersection, interpolation or fill is used.

Public climate examples use statsmodels' bundled data: 732 monthly El Niño values
in Celsius, and 444 monthly CO2 means in ppm from 1965 onward. CO2 monthly means
use finite weekly measurements, with at least one in every retained month. There
is no future interpolation. Climate stays in physical levels; no log Celsius
transform is applied. Period ends are an observational availability convention;
actual historical publication dates and vintages are unknown. Climate horizons
are monthly observations, not stock sessions or trading-return labels.

## Model and selection contracts

The frozen menu is AR(5) on smoothed differences, Ridge(alpha=10), histogram boosting,
GRU and NBEATS_demo. Climate adds period-12 seasonal EMA-change repetition with
exact EMA inversion to actual changes. Nulls are always eligible. Drift/slope
comparators are reported but not admitted to selection.

The learned models retain the 20-by-2 EMA-change features (.2/.05) and at most 252
mature examples. Each track/dataset gets separate fits. AR(5) uses its native
smoothed forecast plus the earlier restricted historical actual-change bridge;
this is a complete-procedure comparison, not an architecture-only experiment.
Neural models retain three fixed seeds, width 12, 30-epoch ceiling, patience 5,
purged chronological stopping and reset/refit; family forecasts average seeds.
A failed seed triggers the entire family's recorded null fallback.

Before each outer block, selection uses equal-horizon relative MAE from earlier
inner origins with all five labels known. All outer choices remain frozen within
the block; parameters refit at every origin. Stocks have six inner origins per
block spaced 20 sessions apart; climate has four spaced ten months apart. Outer
starts/steps are recorded in the protocol and notebook. Only complete five-horizon
cohorts are scored, plus one latest wholly unscored origin per dataset:

- Stocks: **64** scored origins per asset and horizon, shared across all six assets.
- El Niño: **28** scored origins per horizon.
- CO2: **24** scored origins per horizon.

There are **576 distinct dataset/origin fits** including inner and latest queries,
**6,912 successful neural seed fits**, **2,304 successful neural ensembles**, and
**96 latest-origin neural checkpoints**. Tabular fits and classical track forecasts
have their own complete audits. No failed outer dates are dropped.

## Execution and correctness

**132 tests pass**, including all earlier 120 and twelve new cases covering seasonal
EMA inversion, prefix poisoning, complete outer cohorts, equal-dataset aggregation,
synchronized asset resampling, calendar mismatch rejection, parameter validation
and refusal to reuse edited model-stage caches.

TS09 contains **8 executed code cells, 6 inspected figures and 20 CSV tables**.
Fresh model computation produced the full benchmark before rendering. A shared-
colorbar layout conflict was repaired; the saved notebook was then rebuilt and
executed in a fresh process using the completed model stage bound to exact protocol,
source, result and checkpoint hashes. The final execution has no errors or warning
streams. Cached chart/report execution takes about four seconds; that is not the
full model-fitting runtime. Set `RECOMPUTE=True` in the model cell for fresh fitting.

The artifact verifier reconstructs every dataset's targets and paired scorecards,
checks inner/training cutoffs, selected-copy equality, fallback values, scaler means,
chronological stopping boundaries, three-seed averages and all checkpoint queries.
All 96 checkpoints reload with `weights_only=True` and reproduce saved residual and
decoded predictions exactly in the validated CPU environment. They are inference
snapshots, not resumable optimizer checkpoints. The completed-stage cache's source
and output digests are also reconciled with the execution manifest.

Macro aggregation averages within-dataset skills with equal dataset weights and
separate stock/nonfinancial groups. No pooled Celsius/ppm/log-price error is used.
The stock panel resamples synchronized time blocks across all assets. Per-asset
null-MAE denominators stay fixed in that conditional resampling. A test confirms
that duplicating an identical asset does not falsely narrow the panel ranges.

The 0.6.0 wheel matches all current source modules. The stable ZIP is extracted to
an independent temporary folder, all checksums are verified, and the full saved-
artifact audit runs without the original checkout or prior numbered archives.

## Bounded findings

Five-observation selected-procedure MSE skill versus the dataset-specific null:

| Dataset | Smoothed level | Actual change |
|---|---:|---:|
| SPY | -0.0633 | -0.3585 |
| MCHP | -0.1089 | -0.1370 |
| MRVL | -0.1842 | -0.0674 |
| TXG | 0.0023 | 0.0419 |
| GOOGL | -0.1691 | -0.0161 |
| NVDA | 0.0004 | -0.0535 |
| El Niño, monthly | 0.8476 | 0.8030 |
| CO2, monthly | 0.9844 | 0.9770 |

Equal-asset stock means are **-0.0871** smoothed and **-0.0984** actual return. Only
2/6 and 1/6 asset scores are positive. The two positive smoothed scores are small.
This fixed menu/selector did not beat the nulls on average in this historical
basket. It does not establish that all possible stock forecasters must fail.
Climate predictability does not transfer to stocks or prove state-of-the-art
climate forecasting; the seasonal comparator is also reported.

For four-origin synchronized blocks, normalized selected-minus-null MAE is +0.0256
with range [0.0033,0.0529] smoothed and +0.0254 with range [0.0005,0.0593] actual
return. Positive favors the null. One- and two-origin ranges include zero for both
tracks. These ranges depend on the stated resampling design; they are not bounds
on MSE skill, independent-asset inference or multiplicity-adjusted significance.

## Independence, visual review and next stage

All stock histories were examined elsewhere in the project; the universe is a
convenience basket with related technology names. Climate data are public teaching
series. Computational timing checks pass, but a pristine independent holdout,
point-in-time vintages, completed prospective validation and trading advantage
remain **not established** in the explicit audit table.

The prospective protocol is a specification only, with first eligible issuance
no earlier than 2026-11-02. It starts no collection service or automation and has no
collected outcomes. Deployment acceptance thresholds require separate review.

All six figures were inspected directly: data inventory, per-asset model heatmap,
horizon profiles, climate comparison, chronological-block variation and synchronized
uncertainty. The colorbars were repaired and rechecked. HTML figure counts and alt
text pass programmatic checks. HTML browser layout and MathJax remain unverified
because local file previews were previously rejected by the browser surface.

From `forecasting_library`, with the existing requirements:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS09_Cross_Dataset_and_Cross_Asset_Benchmark.ipynb
python scripts/verify_artifacts.py
```

`BENCHMARK_GUIDE.md` explains the findings. TS10 remains the separate NB9/NB10
integration stage, requiring explicit executable-price and horizon mapping.
