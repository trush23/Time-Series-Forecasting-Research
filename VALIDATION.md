# Milestone 1 validation record

This historical record is preserved. See `VALIDATION_MILESTONE_2.md` for TS03–TS04,
the extended test suite and package version 0.2.0.
See `VALIDATION_MILESTONE_3.md` for TS05–TS06 and package version 0.3.0.

Validation date: 2026-10-31. Scope: TS00–TS02 and the reusable causal-filter,
target/maturity, baseline and diagnostic modules in version 0.1.0.

## Execution and correctness

- **54 unit tests passed** with the existing `/opt/anaconda3/bin/python` environment.
  Exact dependency versions are saved in `environment-validated.txt`.
- **All 20 code cells** executed in order, with one fresh Python/IPython process
  per notebook. No cell computation was skipped or replaced. The in-process helper
  avoids kernel sockets; this validates fresh-process cell execution rather than
  a separate Jupyter-server session. The notebooks also support ordinary Run All.
- Saved artifacts contain **12 embedded figures and 23 CSV tables**, along with
  per-notebook execution records and source/configuration/input/output hashes.
- The pure Python wheel was built without downloading dependencies. Its core
  source files were checked against the validated source tree.
- Artifact verification checks notebook structure, sequential outputs, figure
  presence, manifests, data hashes, common forecast origins and every scored
  historical return against the original snapshot and observed-session index.

The tests verify analytical EMA behavior, pandas EMA equivalence, independent
SciPy IIR references, polynomial least-squares and spline endpoint equivalence,
constant/ramp behavior, Kalman covariance symmetry/positive semidefiniteness,
JSON checkpoint/resume, nonmutating projection, prefix and future-poison
invariance, warm-up, holiday-aware observed-index offsets, delayed-entry labels,
all-horizon maturity, malformed data/specifications and score formulas.

The first pytest invocation encountered an unrelated globally installed napari
plugin during plugin discovery. The project tests then passed with third-party
plugin autoload disabled; the package required no environment modification.

## Visual inspection

All 12 exported figures were inspected directly as rendered images, using
notebook-level contact sheets and full-size checks of the final historical
scorecards. Titles, axes, units, legends, response curves, step overshoot,
revision behavior and the distinction between the two forecast tracks were
reviewed. Stable filter colors are used across figures.

HTML reading copies were generated from the saved executed notebooks. Embedded
figure counts, alt text and observed-result summaries were checked programmatically.
A browser attempt to open the local HTML file was rejected because that browser
surface disallows `file:` URLs. No workaround was attempted. **Browser layout and
MathJax rendering of the HTML copies remain unverified**; figure rendering itself
was inspected directly. This does not affect the executed notebook outputs.

## Bounded findings

- At alpha 0.2, IID observation-noise variance gains are **0.111111 for EMA**,
  **0.253772 for DEMA** and **0.382208 for TEMA**. Lower low-frequency delay
  therefore does not establish equivalent noise suppression.
- In the matched-attenuation demonstration, all six fixed linear filters are
  within **1.60%** of the reference IID-noise variance gain. Endpoint SG uses a
  77-observation degree-two window and has **35.26% step overshoot**. Bessel
  uses a second-order magnitude-normalized design with cutoff about 0.04917
  cycles/session and has **0.489% step overshoot**. These values apply to these
  specified filters, not to all parameterizations of their families.
- Appending observations produces **zero revision in archived causal endpoints**.
  Centered SG and a retrospectively refitted spline curve revise historical values
  in the demonstrated boundary region.
- In TS01, all four simple smooth-slope return candidates have negative
  five-session MSE skill against zero on the previously examined SPY period.
  TS02 likewise retains negative results rather than selecting a favorable
  filter/horizon. These are baseline diagnostics, not evaluations of ARIMA,
  LSTM/GRU or other learned models.
- The Kalman settings are illustrative and not attenuation matched. A single
  latent-signal realization cannot establish a universal denoising winner.

## Reproduction

From this directory, using the environment of your choice with `requirements.txt`:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS00_Foundations_and_Targets.ipynb
python scripts/execute_notebook.py notebooks/TS01_EMA_and_Adaptive_Filters.ipynb
python scripts/execute_notebook.py notebooks/TS02_Endpoint_and_State_Space_Filters.ipynb
python scripts/verify_artifacts.py
```

The notebook builder is supplied for editing/reconstruction; running it replaces
the notebooks with unexecuted cells. Re-execute them afterward. The input-preparation
script requires the original sibling project NB3 folder, but is **not needed** to
run the delivered bundle: its frozen inputs are already included.

## Limits and next milestone

There is no automatic model selection or nested tuning in this milestone. All
market filter settings are fixed without using their forecast losses; TS02's
attenuation matching uses filter mathematics, not market outcomes. Later learned
models require nested chronological selection and an untouched/prospective period.
No P&L, live forecast service, current-data refresh, simultaneous forecast bands
or deployment decision is included. The original NB1–NB10 artifacts are unchanged.


## Final program milestone

TS11 completes TS00–TS11 in package 0.8.0. See
[Milestone 8 validation](VALIDATION_MILESTONE_8.md) for real pinned foundation
checkpoints, both target tracks, preservation, fresh/replay execution and figure
inspection. The complete suite has 163 passing tests and 12 audited executed
notebooks, 95 code cells, 63 figures and 188 CSV outputs.
