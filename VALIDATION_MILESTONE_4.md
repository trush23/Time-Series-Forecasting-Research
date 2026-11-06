# Milestone 4 — TS07 validation

Validated 2026-10-31 with the existing Python 3.12 environment, scikit-learn 1.6.1
and CPU PyTorch 2.5.1. Versions are recorded in
`environment-validated-milestone4.txt`. Package version: **0.4.0**.
No dependency installation, pretrained weights or market-data refresh was required.

## Scope and preservation

TS07 adds five educational feature-to-target adapters: decomposition linear,
last-centered linear, generic N-BEATS-style residual stacks, N-HiTS-style multirate
interpolation and a joint-channel patch transformer. The architecture table in the
notebook explicitly identifies deviations from the published models. These are
not reproductions of their benchmark configurations, interpretability results or
long-horizon performance claims.

All **170 accepted notebook/output files from TS00–TS06** were checked byte-for-byte
against the Milestone 3 archive and remain unchanged. Earlier ZIPs remain unchanged.
The earlier `neural.py` and package initializer are archived under
`provenance/milestone_3_sources/` so old source hashes remain auditable. Existing
compact model code paths and checkpoint schemas remain compatible.

## Execution and checks

TS07 executes independently in a fresh Python/IPython process, with **7 code cells,
6 rendered figures and 18 CSV tables**. Its notebook includes saved outputs and an
observed-results summary; an HTML reading copy is included. All six figures were
inspected directly at readable size: parameter counts, outer skill, frozen matched
procedures, seed variability, training diagnostics and paired MAE ranges.

**107 tests pass**, comprising the previous 98 and nine new cases covering:

- Exact repeated-seed fits, inference-checkpoint restoration and forecast equality
  after changing only observations beyond the forecast origin, for all five models.
- The independent decomposition-linear identity when trend/remainder maps agree,
  and the last-centered adapter's response to an additive channel shift.
- Explicit residual backcast recursion and summed forecast identity for both stacks.
- Patch model batch independence, input ordering sensitivity and shape rejection.

The saved-artifact verifier independently checks labels, complete seed averages,
training/scaler cutoffs, chronological purging, frozen choices, fallback values,
reference digests, source/config/output hashes and every saved checkpoint. The new
**30 latest-origin inference checkpoints** reproduce saved sequences, residual
predictions and decoded predictions exactly after `weights_only=True` loading in
the validated CPU environment. They are inference snapshots, not training resumptions.

The 0.4.0 wheel is built offline from local sources and checked against all current
source modules. The cumulative bundle includes prior historical source snapshots,
all inputs and all frozen references. Extracted-bundle checksums and saved-artifact
verification pass without the original checkout or prior archives.

HTML embedded figure counts and alt text are checked programmatically. HTML browser
layout and MathJax rendering remain unverified: local file previews were previously
rejected by the browser surface. Actual exported figures were inspected directly.

## Model and experiment contracts

Parameter counts are 213, 108, 3,699, 3,021 and 1,709 in the architecture order above.
All use a 20-by-2 window of fast/slow causal EMA first differences (rates .2/.05).
Track A learns the residual future smoothed log level around the exact filter-aware
unchanged-price anchor. Track B learns actual cumulative log returns. Their target
scalers, fits, selection and scorecards remain separate and equally represented.

Decomposition uses a width-5 moving average inside the already observed window;
its centered within-window operation never crosses the forecast origin. Residual
stacks use three blocks with two width-12 hidden layers. Multirate blocks use
max-pooling factors 4/2/1 and forecast knots 1/3/5, interpolated to five outputs.
The transformer uses four nonoverlapping length-5 joint-channel patches, learned
positions, width 12, two attention heads, one encoder and zero dropout. It is not
channel-independent PatchTST and has no future tokens, pretraining or RevIN.

The TS06 budget is retained: at most 252 mature examples, latest 20% for chronological
stopping, five purged origins at the full budget (196 optimization and 51 validation),
30-epoch ceiling, patience 5, full-batch AdamW, learning rate .005, decay .001 and
clipped gradient norm 1. Stopping scalers use optimization data only; epoch choice
is followed by a same-seed reset and refit on all currently mature examples. Equal
epoch ceilings do not imply equal compute. Three fixed seeds (1731/1732/1733) are
averaged; no best seed is promoted. All 1,200 stock seed fits and 400 architecture
ensembles succeeded, including inner and latest unscored origins. Toy fits are
additional in-sample optimization diagnostics, not forecasting evidence.

Selection uses the same three outer blocks, three earlier inner origins per block
and 40-session spacing as TS06. There are 30 scored outer origins per horizon, plus
one latest unscored origin. Equal-horizon relative MAE selects separately by track
from the five architectures plus the null baseline. Drift benchmarks are reported
but not selection candidates. Their exclusion is explicit; beating the null need
not beat drift. All choices are frozen before the relevant outer block.

## Bounded findings

On the previously examined adjusted SPY snapshot ending 2026-10-01:

| Selected procedure | Smoothed five-session RMSE, log bps | Smoothed MSE skill | Actual-return five-session RMSE, log bps | Actual-return MSE skill |
|---|---:|---:|---:|---:|
| TS06 | 91.29 | 0.1655 | 195.30 | 0.0262 |
| TS07 | 93.03 | 0.1333 | 190.17 | 0.0767 |

TS07 improves actual-return MSE in this comparison but has worse smoothed-level MSE
than TS06. Its smoothed MAE is slightly lower (72.92 versus 73.52 log bps), illustrating
different MAE/MSE rankings. The TS07 selector uses the patch transformer in the
first block for both tracks; in block two it uses the patch transformer for smoothed
levels and NHITS for returns; in block three it uses NBEATS for both tracks.

The fixed NBEATS demonstration's five-session MSE skills are 0.2039/0.1494; these
are retrospective fixed-family results and do not replace the selected procedure.
Five-session drift benchmark skills are 0.1882 for filter drift and 0.1283 for raw
drift, both above the corresponding selected TS07 score.

Selected-minus-baseline MAE moving-block 95% percentile ranges include zero for
both tracks at every tested block length (1/2/4 scheduled origins). With blocks of
four origins: smoothed mean -8.83 log bps, range [-18.94, 4.76]; actual-return mean
-7.43 log bps, range [-18.35, 11.39]. These are not intervals for MSE skill or future
returns, and do not correct for prior project exploration or architecture searches.
One previously examined asset and 30 sparse origins do not establish a general
forecasting or trading advantage. Costs and executable trade decisions are not tested.

## Reproduction

From the library directory, with the full requirements installed:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS07_Advanced_Architecture_Demonstrations.ipynb
python scripts/verify_artifacts.py
```

`build_advanced_notebook.py` generates TS07 only as unexecuted cells. It never rewrites
accepted notebooks. Frozen references are already bundled; ordinary reruns should
not refresh them. Different library versions/hardware may change seeded results.
`PERFORMANCE_GUIDE.md` explains the score definitions and matched-date findings.
TS08 probabilistic forecasting and chronological calibration remain the next stage.
