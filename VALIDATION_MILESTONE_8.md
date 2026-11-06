# TS11 — Foundation-model laboratory validation

Validated **2026-11-01**, package **0.8.0**. All twelve planned notebooks,
**TS00–TS11**, are complete. TS11 adds one real pretrained model family at two
compact sizes; it does not claim to cover every foundation-model family.

## Executed scope

- **163 tests passed**, including 19 new algebra, boundary, maturity and
  future-label-contamination cases. The full suite also passes in the independent
  portable copy.
- TS11: **10 executed cells, 6 directly inspected figures, 17 CSV outputs**.
- Entire suite artifact audit: **12 notebooks, 95 executed cells, 63 figures,
  188 CSV outputs**. Earlier accepted notebooks are audited, not rerun unnecessarily.
- **38,880 forecast rows**: eight datasets × 81 origins × six procedures ×
  two tracks × five horizons. The 81 origins comprise 40 calibration, 40 evaluation
  and one wholly unscored latest origin. There are **19,200 evaluation rows** and
  **480 latest unscored rows**.
- **2,592 dataset/context inferences**, plus 256 independently generated control
  histories across the two checkpoints. All 34 raw output arrays are retained.
- **1,296 mature Ridge fits**, **2,592 context audits**, **3,936 interval-calibration
  audits**, **19,680 residual interval rows**. No model fallback or forecast exclusion.
- No native quantile crossings occurred in the retained financial/climate forecasts.

Fresh real-checkpoint inference was executed offline in an isolated environment
using the existing Python 3.12.2 scientific stack, torch 2.5.1, Chronos 2.3.2,
transformers 4.51.2, accelerate 1.15.0 and huggingface-hub 0.30.2. The exact dependency
record is `environment-validated-milestone8.txt`. We installed only Chronos and
accelerate in that temporary environment; the user's base environment was unchanged.
The upstream transformer emitted a legacy-cache deprecation notice during fresh
inference. It did not cause numerical or execution failure.

The final delivered notebook was then executed in the original project environment
with **no Chronos installation**, using its default audited inference replay.
Ridge fits, adapters, labels, scores, intervals and figures were freshly computed.
Fresh inference took about 20 seconds for the full notebook; default replay took
about 12 seconds in the workspace, and 13 seconds in the independent portable copy.
These end-to-end readings are local observations, not benchmark guarantees.

The temporary environment inherited Python packages but not Jupyter's HTML template
directory. Setting `JUPYTER_PATH` to the parent environment's `share/jupyter`
resolved that export-environment issue; all final HTML exports succeeded.

## Frozen protocol and source preservation

`TS11_PROTOCOL.json` SHA256:
`073a1cf2d069fc9eaab21c257c7f21d9b2763439f94c7bf652c9f9c5179517d6`.
The initial pre-score experiment declaration is retained under
`provenance/TS11_protocol_initial.json`, SHA256
`bfdb724bbb4e0d830015bbf1cde2113771bc9a12066ff25aec31f363452d6551`.
After initial execution only the CSV baseline identifier changed from `null` to
`filter_null`, avoiding pandas' default missing-value interpretation. No cohort,
parameter, candidate, forecast value, metric or selection changed in that repair.

Exact tiny/mini commit revisions, downloaded model cards, configs, weights, licenses
and per-file digests are bundled. Actual parameter counts are **8,652,672** and
**21,236,096**; weight sizes are **34,622,352** and **84,956,096 bytes**.
The inference archive records source, effective protocol, input-signal, checkpoint
manifest and cache hashes. Replay rejects changed provenance. Neural dependencies
remain optional and are imported only for explicit fresh inference.

All **434 existing notebook/output files** in the maintained user copy were checked
byte-for-byte against the staged package and preserved. The user's TS10 rerun had
unchanged cell sources and numerical outputs, increasing execution counts
1/3/5/7/9/11/13/15, and only a refreshed manifest creation timestamp; those exact
notebook/manifest bytes were adopted. Other prior outputs remain unchanged.
Archived execution source versions preserve earlier notebook provenance after the
package's version-only initializer update. No synced `sources/` material changed.

## Independent checks

`verify_foundation.py` independently reconstructs:

- checkpoint bytes and inference identity;
- all scored actual-level and actual-change labels and observed-step endpoints;
- every retained context digest and its final information boundary;
- every native .5 point path and its EMA-inverted actual-change forecast;
- every point MAE/MSE and relative skill, and every equal-dataset macro score;
- all residual-calibrated bounds/medians from the correctly matured earlier vectors;
- interval coverage, width and interval score;
- wholly unscored latest rows and known-null control bounds.

Prefix replay and later-value perturbation leave the tested SPY contexts exactly
unchanged. All Ridge training-label endpoints and interval-calibration-label
endpoints are no later than the forecast origin. Tests poison not-yet-mature labels
and confirm that earlier calibration bounds are unaffected. Analytic tests recover
known actual paths from their EMA paths over four alpha values, and confirm zero
actual change under the filter-catch-up path.

The full artifact verifier checks notebook structure, increasing execution counts,
error-free cells, embedded/exported figure counts, HTML alternative text, source
and output digests, all prior milestones and the specialized TS11 audit. The 0.8.0
wheel's module bytes were checked against every current source module.

## Visual inspection

All six final TS11 figures were directly viewed in a contact sheet, with readable
axes/legends and no clipped labels: macro skill, stock heterogeneity, residual
coverage, paired ranges, martingale control and compute cost. Heatmap annotations
are rounded percentage skills; paired ranges explicitly use negative-is-better
normalized MAE differences. Nominal coverage and known-null reference lines are
visible. HTML embeds the same figures with descriptive alternative text.

## Bounded findings

Primary five-step stock MAE skill, equal weighting over six stocks with forty
origins each:

| Procedure | Smoothed | Actual returns |
|---|---:|---:|
| Fixed Ridge_10 | -6.33% | -2.49% |
| Tiny, context 256 | -26.07% | -12.60% |
| Mini, context 256 | -25.87% | -10.31% |

Neither foundation checkpoint beats the matched null in this experiment. Both
primary checkpoints have zero positive-skill stock datasets at h=5 for either
track. The six-stock origin cohort is 2025-12-12 through 2026-09-24; the data snapshot
ends 2026-10-01. Those are historical, previously examined adjusted-price vintages,
not real-time prospective forecasts. Climate results can be positive and remain
separate in physical units and monthly horizons.

Nominal 80% residual intervals cover roughly 75–77% of the stock outcomes. Width,
interval score and pinball losses accompany coverage. No finite-sample coverage
guarantee is asserted. Paired block ranges at 1/4/8 origins are exploratory, with
synchronized stock resampling, fixed per-asset normalizers and no multiplicity
adjustment. Point .5-path inversion is not a claim about a conditional return
median, and marginal quantiles are never differenced into a return distribution.

Checkpoint training end dates and exact exposure to each evaluation history are
unknown. Chronological causality checks do not establish pretraining independence.
No fine-tuning, score-driven checkpoint selection, trading promotion or order occurs.

## Reproduction

Open TS11 with the existing notebook kernel and run top-to-bottom, or use:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS11_Foundation_Model_Laboratory.ipynb
python scripts/verify_artifacts.py
```

For fresh checkpoint inference install the optional pinned requirements and change
`RECOMPUTE` to `True`. All weights are local; inference itself requires no network.
The online preparation script is separate and must not be used to silently update
the pinned checkpoint/protocol. Maintained delivery paths retain their existing
names under `Documents/Trading/Codex`.
