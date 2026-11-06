# Reading TS05–TS07 performance

Start with the **selected procedure**, then inspect families, seeds and diagnostics.
These numbers describe a previously examined adjusted SPY snapshot ending
2026-10-01. They do not estimate trading profits or establish a market advantage.

## Two different questions

- **Smoothed level:** how well can we forecast the future causal EMA log level?
  Its baseline propagates an unchanged raw price through the EMA, accounting for
  filter catch-up. Plain smoothed-level persistence is a weaker comparison.
- **Actual return:** how well do smoothed inputs predict the actual cumulative
  future log return? Its baseline is zero return. Smoothing the predictor does not
  smooth this target. Each track uses its own separately fitted models.

Both tracks use horizons of 1–5 **observed sessions**, not calendar days. A score
from one track cannot be ranked against a score from the other as if the targets
and denominators were identical. Actual returns remain the relevant target for
subsequent economic evaluation; a smoother's predictability can include its own
mechanical response.

## Metrics and units

MAE is mean absolute error. RMSE is the square root of mean squared error and puts
more weight on large misses. Smaller is better for both.

`MSE skill = 1 - model MSE / baseline MSE`, evaluated on identical origins.

- Skill 0: same MSE as the baseline.
- Skill +0.20: 20% lower MSE; RMSE is `sqrt(0.8) = 0.8944` times baseline RMSE,
  or 10.56% lower.
- Skill -0.20: 20% higher MSE; RMSE is `sqrt(1.2) = 1.0954` times baseline RMSE.

A skill of 0.20 is not a 20% return or a 20% directional success rate. Error in
**log basis points** is log error times 10,000, not exact arithmetic-return basis
points. Smoothed log-level error can be measured in these units without being a
forecast of a tradable raw price.

## Families, seeds and selected procedures

A family score evaluates one fixed model specification at every outer origin.
Neural family forecasts average three predeclared seeds. A seed dot measures
initialization sensitivity on the same data; it is not an independent sample of
markets or a confidence interval. The best seed is never chosen.

The selected procedure chooses a family or the declared track baseline using
**earlier inner data**, separately by track and before each outer block. It then
freezes that choice for the block. It can underperform a fixed family identified
in hindsight. Promoting the best outer family would turn the outer evaluation
into another selection sample.

Training and stopping minimize standardized MSE. Selection minimizes equal-horizon
relative MAE. Reported MSE skill uses unstandardized outer losses. These objectives
can disagree. A low standardized validation loss need not imply low future loss,
especially when volatility changes. Only three inner origins per block determine
selection here, so choices can be noisy.

Drift benchmarks are reported but are **not eligible selection candidates** in
TS05–TS07. A positive score against the null therefore need not beat drift.
Training budgets match TS05–TS07 in examples and origin schedule; neural fits
share an epoch ceiling, not equal runtime or equal parameter count. TS04 also
uses different training/selection budgets and a restricted return bridge.

## Five-session comparison of frozen selected procedures

All rows below use the same 30 scored origins and targets. Prior block choices
remain frozen; no earlier notebook was rerun or reselected for this comparison.

| Procedure | Smoothed RMSE, log bps | Smoothed MSE skill | Actual-return RMSE, log bps | Actual-return MSE skill |
|---|---:|---:|---:|---:|
| TS04 classical | 116.70 | -0.3638 | 195.30 | 0.0262 |
| TS05 tabular | 103.27 | -0.0679 | 203.83 | -0.0606 |
| TS06 compact neural | 91.29 | 0.1655 | 195.30 | 0.0262 |
| TS07 adapted architectures | 93.03 | 0.1333 | 190.17 | 0.0767 |
| Null baseline | 99.93 | 0 | 197.91 | 0 |

TS07 improves actual-return MSE relative to TS06 in this sample, while TS06 has
lower smoothed-level MSE. TS07 has slightly lower smoothed MAE than TS06 despite
its higher smoothed RMSE: 72.92 versus 73.52 log bps. This is a concrete example of
MAE and MSE rankings disagreeing.

TS07's fixed NBEATS demonstration has five-session MSE skills of 0.2039 for
smoothed levels and 0.1494 for actual returns. Those are family results, not the
selected-procedure results above. Its higher retrospective score is not permission
to replace past procedure choices.

Simple drift benchmarks also deserve attention: five-session MSE skill is 0.1882
for filter drift and 0.1283 for raw drift. Both exceed the corresponding TS07
selected-procedure score. Their inclusion is diagnostic; these comparisons do
not prove that any model or drift forecast will retain an advantage prospectively.

## What the uncertainty ranges say

The saved moving-block 95% percentile ranges compare selected-minus-baseline
**mean absolute loss at horizon five**. They are not ranges for MSE skill or for
the next realized return. Blocks count scheduled origins (40 sessions apart),
not trading days.

For TS07, blocks of four origins give:

- Smoothed: mean difference -8.83 log bps; range [-18.94, 4.76].
- Actual returns: mean difference -7.43 log bps; range [-18.35, 11.39].

Both include zero; so do the other two tested TS07 block lengths. These conditional
resampling ranges do not account for prior exploration, multiple architecture
searches, asset selection or arbitrary nonstationarity. Thirty sparse observations
from one asset provide limited evidence even when a point estimate is positive.

Use the full horizon curves and saved scorecards rather than only the fifth day.
TS08 will treat forecast uncertainty and chronological calibration separately.

Evidence: `outputs/TS07_Advanced_Architecture_Demonstrations/` contains
`matched_prior_pipeline_comparison.csv`, `performance_interpretation.csv`,
`seed_scorecards.csv`, `selections.csv` and `selected_paired_uncertainty.csv`.
