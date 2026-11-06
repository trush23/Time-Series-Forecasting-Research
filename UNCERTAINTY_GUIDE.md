# Reading TS08 forecast uncertainty

A forecast interval describes plausible **future target values** under a forecast
procedure. It is different from the moving-block ranges in TS05–TS07, which describe
uncertainty in a **historical mean loss difference**. Neither is a guarantee for
an individual future observation.

## Four questions to ask

1. **Coverage:** how often did the outcome fall inside the interval? For a nominal
   80% interval, roughly 80% is the intended long-run frequency under calibration
   assumptions, not a claim of certainty for a given forecast.
2. **Width:** how wide was the interval? Very wide intervals can cover nearly
   everything and provide little useful resolution.
3. **Proper score:** does the forecast balance sharpness and misses well? WIS,
   pinball loss and CRPS address this. Smaller loss is better. Skill is
   `1 - model loss / paired baseline loss`, so positive skill favors the model.
4. **Joint coverage:** did all five outcomes lie inside the five bounds together?
   Five individually calibrated intervals are not automatically a calibrated
   five-horizon band.

## What TS08 evaluates

Both tracks retain equal priority: future causal EMA log levels and actual future
cumulative log returns, at horizons 1–5 observed sessions. The point model is fixed
Ridge_10, alongside the corresponding null for each track. Ridge is a teaching
choice, not selected as the best previous model. It uses the same 20-by-2 history
of EMA changes and at most 252 mature training examples as before.

Forecasts are issued daily from position 650. Calibration uses the last 126 issued
forecast-error vectors whose full five-session outcomes are known. For a forecast
at t, the newest eligible calibration origin is t-5. These are genuine earlier
forecast errors, not residuals calculated after refitting on their outcomes.

The scored cohort has **239 origins per horizon**, five sessions apart, plus the
latest wholly unscored snapshot origin. Earlier TS04–TS07 summaries used 30 origins;
do not compare the numerical scores directly without matching origins.

Gaussian and empirical residual distributions produce seven quantiles. Symmetric
absolute-error and volatility-scaled rank intervals use the corrected rank
`ceil((n+1)*coverage)`. A separate max-normalized-error construction targets a
rectangle containing all five horizons. These are rolling demonstrations; the
exchangeability conditions for ordinary split-conformal guarantees do not hold
by construction for these refitted, dependent stock forecasts. No adaptive online
conformal guarantee is claimed.

## Main five-session findings

For Ridge with 126-error volatility-scaled calibration:

| Track | Nominal marginal coverage | Actual marginal coverage | Mean width, log bps | WIS skill versus same-method null |
|---|---:|---:|---:|---:|
| Smoothed level | 80% | 80.8% | 345.0 | -0.1464 |
| Actual return | 80% | 77.8% | 691.8 | -0.1188 |

The null distributions score better despite the reasonably close coverage. The
negative skill means Ridge's mean WIS is about 14.6% higher for smoothed levels
and 11.9% higher for actual returns. Good coverage alone does not show useful
predictive information or a trading advantage.

For the same nominal 80% setting, joint coverage across all five outcomes is:

| Track | Five marginal scaled intervals | Separately calibrated joint-max band |
|---|---:|---:|
| Smoothed level | 67.8% | 80.8% |
| Actual return | 56.1% | 78.7% |

The joint bands are wider. Their scalar per-horizon interval score/WIS/CRPS is
intentionally missing: a proper joint-distribution score has not been supplied.
For marginal rank methods, WIS is available but CRPS is missing because no full
CDF has been specified. A missing score is not zero.

## Simulation and sensitivity

The independent Gaussian control averages 200 replicates, each with 126 calibration
vectors and 500 test vectors. Corrected 95% absolute-rank intervals attain about
95.3% marginal coverage. When test volatility suddenly triples but calibration
stays unchanged, coverage falls to about 49.8% for smoothed errors and 49.7% for
returns. Oracle rescaling restores the control behavior; the stock-data scale
proxy is not an oracle.

The stock proxy uses only past smoothed changes. Doubling calibration history to
252 daily vectors changes Ridge's five-session scaled 80% coverage to 80.8% for both
tracks; its widths become 328.0 and 654.2 log bps. This is sensitivity evidence,
not an outer-score rule for selecting that window. Local rolling coverage varies
materially even when the aggregate coverage is near its nominal value.

The 126-atom empirical distribution has only about 3.15 observations in a 2.5%
tail. Tail estimates are coarse; serial dependence further limits information.
Latest residual simulations resample intact five-error vectors, preserving their
within-vector dependence. They are not simulated future stock-price processes.
The latest origin is the historical snapshot date **2026-10-01**, not today's date.

## Where to inspect the evidence

Open TS08 beside its saved output tables:

- `interval_scorecard.csv`: coverage, width and interval score by model/method/level/horizon.
- `distribution_skill.csv`: Ridge versus its paired same-method null.
- `joint_coverage.csv`: the event that all five targets lie inside the bounds.
- `quantile_scorecard.csv`: pinball losses for each quantile.
- `calibration_window_sensitivity.csv`: both declared history lengths.
- `calibration_audit.csv`: every calibration cutoff and sample count.

The previously examined single-asset snapshot, overlapping residuals and shared
training/calibration windows limit statistical interpretation. TS09 will address
cross-dataset and cross-asset evaluation; prospective testing remains necessary
before promoting a research demonstration.
