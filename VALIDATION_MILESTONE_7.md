# TS10 — NB9/NB10 integration validation

Validated **2026-11-01**, package **0.7.0**, with the existing Python 3.12,
NumPy/pandas/SciPy/scikit-learn environment. Exact versions are recorded in
`environment-validated-milestone7.txt`. No dependency installation, pretrained
model, market-data download or trading connection was required.

## Scope, frozen references and preservation

TS10 integrates both native forecast tracks with the NB9 next-close target clock,
replays the original NB9 forecasts, reproduces NB10's reference trading results,
and compares fixed Ridge_10 joint residual distributions under the inherited rule.
Canonical NB9/NB10 notebooks and output files remain unchanged. Thirty-seven exact
reference files, including matching NB3 inputs and original notebook copies, are
bundled under `inputs/nb_integration` and independently checked by byte digest.

The market CSV SHA256 is
`4512e193419930f199f6deecb9ae769d1a485128aa87529afee83fb379e69eb3`.
It is the exact NB9/NB10 adjusted SPY snapshot through 2026-10-01, not silently
interchanged with another TS-series snapshot. `TS10_PROTOCOL.json` was frozen
before new scores, SHA256:
`6721b39b3d3e0d298771d09d9a26173f24a6955d0118a183e544a9fd98d5bfce`.

The maintained folder and ZIP in Documents/Trading/Codex retain their stable names.
Existing TS00–TS09 notebooks and outputs match the user's current maintained copy,
including the reexecuted TS09 notebook and its refreshed manifest timestamp. Its
cell sources and numerical outputs were unchanged; increasing Jupyter counts are
retained without normalization. The publisher accepts those edits only because
the staged package preserves the same edited bytes, and otherwise rejects conflicts.
No synced `sources/` file was modified.

## Timing and statistical contracts

Native TS forecasts have five session steps from signal t. NB9/NB10 enter at
close t+1 and exit at t+1+h, so compatible holding horizons are **1–4**. Legacy
h=5 needs six steps and remains a separate reference. The primary holding horizon
is inherited h=3, offset zero; no favorable horizon is promoted from sensitivity.

For a joint native vector R[j], delayed actual return is R[h+1]-R[1]. For native
smoothed levels S[j], the derived target is smoothed change S[h+1]-S[1]. Samples
are differenced before quantiling. A counterexample and unit test show why marginal
quantiles cannot be subtracted. Trading accepts only the actual-return track.

Fixed direct Ridge_10 uses the established 20-by-2 smoothed-change input and at
most 252 mature examples, separately per track. There are **3,098 successful fits**
and **30,980 native issued-forecast ledger rows**. Residual distributions use the
latest 126 earlier complete forecast-error vectors with origin+5 <= query. They
are prequential errors, not in-sample residuals. No stock-data coverage guarantee
is asserted. TS_null_empirical can have a nonzero residual-corrected mean and is
not the zero-return point control.

Both new methods share the original NB9 support overlay; Ridge additionally
requires current fit availability. This is a controlled comparison overlay, not
an intrinsic ridge support detector. The legacy analog uses raw-return/volatility/
volume coordinates, unlike the smoothed-input ridge mean model. Raw historical
volatility remains a risk-sizing overlay. The comparison is between complete
procedures, not an isolated architecture or feature experiment.

There are **654 matched scored origins per holding horizon** for both delayed
tracks, **2,620 calibration audits**, **26,184 delayed forecast rows** (including
24 latest immature controls/distributions), and **2,016 latest delayed samples**.
The primary schedule has 218 opportunities. Its common wealth calendar is
**2024-02-14 through 2026-09-25**, 656 closes/655 daily return intervals. It ends
before the snapshot because the original NB9 cohort requires all five delayed
holding outcomes to have matured.

## Execution and correctness

The notebook runs independently, top-to-bottom in a fresh Python/IPython process:
**8 executed cells, 5 inspected figures, 34 CSV tables**, no errors or warning
streams, about 18 seconds. The HTML reading copy contains the executed figures,
tables and data-derived summary. Browser layout and MathJax remain unverified
because local-file previews were previously rejected; rendered figures were
inspected directly.

**144 tests pass**, comprising the previous 132 and twelve new cases covering:

- Joint-sample subtraction, a marginal-quantile counterexample and shape rejection.
- Exact cash/share account factors at zero/5/20 bps, fee reconciliation, compounded
  wealth, zero-position cash, sell-before-buy rolls and overlap rejection.
- Entry/exit offsets, native horizon-budget rejection, and rejection of smoothed
  targets as trading inputs.
- Decision-weight immunity to poisoned actual outcomes; legacy neighbor label
  maturity/separation and forecast replay; immature-error poisoning in the bridge.
- Risk/cost configuration rejection and audited TS09 cache compatibility after a
  version-only initializer change, while rejecting substantive initializer edits.

The saved-artifact audit independently reconstructs native/delayed actual targets,
training pools/scalers, every new sample-distribution mean/q10/q90/CRPS, all legacy
NB9 forecast fields, NB10 ledger identities, original daily ledgers and metrics,
causal volatility/weights, latest immaturity and frozen cost monotonicity.

All **9,810 original NB9 rows** are reproduced. Mean, q10, median, q90, p_up and
actual values match exactly in the validated environment. Maximum CRPS discrepancy
is about **8.3e-17**. Original NB10 analog/recent/buy-and-hold/cash daily ledgers
reproduce within **1.4e-9 dollars**; original metrics, cost grid and all 15 original
horizon/offset cases also reconcile. The new comparison reports all 40 combinations
of four methods, holding horizons 1–4 and their offsets.

TS09's completed-stage cache remains reusable when all recorded model dependencies
are unchanged. A verified version-only initializer change and unrelated added
modules no longer force refitting. Its original source/output provenance remains
recorded; changed model code, protocol or results still reject reuse. Source snapshots
and content-addressed fallback files preserve historical executions across reruns.

The 0.7.0 wheel matches all 25 current source modules. The stable ZIP is extracted
into an independent temporary directory, all checksums are verified, and the full
artifact audit runs without the original checkout or earlier archives. The old
TS09 cache and new package imports are also checked in that extracted copy.

## Ledger and interpretation limits

The inherited rule uses next-close fills, 5 bps assumed cost per side, a five-log-bps
buffer, 10% volatility target, 1% tail budget and at most 100% long exposure. Cash
earns zero; no shorting or leverage is enabled. Entry quantities account for fees
inside the budget; exits deduct fees from proceeds. Allocation fractions and
eligibility stay fixed in cost sensitivity, while subsequent quantities can change
with account equity. Tail sizing is heuristic, not a guaranteed loss bound.

Adjusted closing prices and fractional research units are not executable historical
quotes or guaranteed auction access. Slippage/impact, taxes, publication/vintage
issues and intraday risk are not reconstructed. Drawdown is marked at closes.
The mean-log threshold is not a significance test or expected-profit guarantee.

Primary hypothetical results over the common calendar:

| Rule/distribution | Net total return | Maximum drawdown | Mean exposure | Trades |
|---|---:|---:|---:|---:|
| Legacy analog | 4.49% | -9.74% | 27.4% | 100 |
| Legacy recent | 0.99% | -15.01% | 54.6% | 194 |
| TS ridge empirical | 1.07% | -10.17% | 44.7% | 134 |
| TS null empirical | 3.87% | -7.35% | 52.4% | 166 |
| Buy-and-hold | 58.52% | -18.76% | 99.8% | 1 |
| Cash | 0.00% | 0.00% | 0.0% | 0 |

These are total returns, not annual returns or risk-adjusted alpha. All tested
paired strategy-versus-recent mean daily return ranges include zero at block
lengths 10/20/40. For ridge, the point difference is about 0.0011 bps/day; the
20-session range is [-2.55,2.73] bps/day. Ranges do not correct for prior exploration.
At h=3, ridge's actual-return and smoothed-change MSE skills are -0.2390/-0.2668
against the corresponding deterministic point controls. Positive historical wealth
does not establish superior forecasts or an independently validated trading edge.

`research_only_latest_handoff.csv` contains eight latest ridge records: both tracks
and holding horizons 1–4, all marked `research_only_no_order` and immature. The origin
is 2026-10-01. No future fill price/label is supplied, no live NB10 input is overwritten,
and no service, trade or automation is started.

## Reproduction and next stage

From the library directory with its existing requirements:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/execute_notebook.py notebooks/TS10_NB9_NB10_Forecast_and_Trading_Integration.ipynb
python scripts/verify_artifacts.py
```

`build_integration_notebook.py` generates TS10 only. Frozen references are already
bundled; ordinary reruns do not refresh them. `TRADING_INTEGRATION_GUIDE.md` explains
the results. TS11 remains the optional foundation-model laboratory. Prospective
forecast collection and execution-grade acceptance remain separate work.
