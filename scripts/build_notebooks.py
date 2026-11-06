"""Generate the three milestone notebooks using nbformat."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/"notebooks"
DEST.mkdir(exist_ok=True)


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


SETUP = '''
from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src'/'tslab').is_dir())
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from tslab.smoothing import FilterSpec, CausalSmoother, smooth
from tslab.data import synthetic_signal, load_snapshot
from tslab.targets import make_targets, mature_labels
from tslab.evaluation import benchmark, scorecard, paired_block_difference
from tslab.diagnostics import audit_replay, response_metrics, latent_scores, impulse_response
from tslab.reporting import configure_plots, save_figure, write_manifest, filter_configs
configure_plots()
COLORS = dict(zip(['Unfiltered','EMA','DEMA','TEMA','KAMA','Butterworth','Bessel',
                  'Endpoint SG','Weighted local linear','Trailing spline','Kalman (unmatched)'],
                 ['#777777','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9',
                  '#AA4499','#332288','#882255','#117733']))
SEED = 1731
OUT = ROOT/'outputs'/'{name}'
OUT.mkdir(parents=True, exist_ok=True)
'''


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "milestone": 1, "research_tracks": ["smoothed", "returns"]})
    nbf.validate(nb)
    nbf.write(nb, DEST/f"{name}.ipynb")


save("TS00_Foundations_and_Targets", [
md(r'''
# TS00 — Foundations, targets, and negative controls

## Goal
Define two equal-priority forecast tracks over **1–5 observed sessions**:
future causal smoothed log levels and future actual log returns using smoothed inputs.
Before comparing sophisticated models, demonstrate the predictability introduced by a filter.
This is a methods laboratory, not evidence of trading profitability.
'''),
md('''
## Setup
Run with a Python environment containing the dependencies in `requirements.txt`.
No download, credentials, earlier notebook execution, or hidden kernel state is required.
All figures and tables are saved in `outputs/TS00_Foundations_and_Targets`.
'''), code(SETUP.format(name="TS00_Foundations_and_Targets")),
md(r'''
## Steps
### 1. Define the data and targets
We begin with a known trend/cycle and IID observation noise. Synthetic weekday
timestamps are an observation schedule, not an exchange calendar. Session horizons
advance through the index supplied by the data source; there is no calendar filling.

For $x_t=\log P_t$, the two labels are $s_{t+h}$ and $x_{t+h}-x_t$.
The smoothed label at $t+h$ is the endpoint issued using information only through $t+h$.
'''), code('''
demo = synthetic_signal(seed=SEED)
spec = FilterSpec(kind='ema', alpha=.2, warmup=60)
x = demo.observed.rename('log_price')
s = pd.Series(smooth(x, spec), index=x.index, name='smoothed')
targets = make_targets(x, s)
as_of = x.index[400]
eligible = mature_labels(targets, as_of, require_all_horizons=True)
assert (eligible.available_at <= as_of).all()
assert eligible.groupby(['track','origin']).horizon.nunique().eq(5).all()
targets.to_csv(OUT/'targets.csv', index=False)
eligible.tail(10).to_csv(OUT/'mature_label_example.csv', index=False)
display(eligible.tail(10)[['origin','track','horizon','target_end','actual']].round(6))
fig, ax = plt.subplots()
ax.plot(demo.index, demo.observed, color='.7', lw=.7, label='Noisy synthetic observation')
ax.plot(demo.index, demo.latent, color='black', ls='--', label='Known latent signal')
ax.plot(s.index, s, label='Causal EMA (alpha=0.2)')
ax.set(title='Known latent signal versus an operational smoothed target', ylabel='Log level', xlabel='Synthetic observation date')
ax.legend(ncol=1)
save_figure(fig, OUT, '01_target_definition');
'''),
md(r'''
The filtered series is an operational target, not latent truth. Its warm-up remains
missing. The last five-session labels remain unavailable until their end timestamps.

### 2. Derive and test a filter-aware null
Assume $x_t$ is a martingale. For an EMA with $q=1-\alpha$,
$$E[s_{t+h}\mid\mathcal F_t]=q^h s_t+(1-q^h)x_t.$$
Under independent future returns of variance $\sigma^2$, its conditional forecast MSE is
$$\sigma^2\sum_{j=1}^h(1-q^j)^2.$$
Holding the smoothed value constant adds the squared deterministic catch-up term.
These identities describe the assumed null, not a market law.
'''), code('''
rng = np.random.default_rng(SEED)
draws, sigma, current_x, current_s = 30000, .01, 1., .95
future_x = current_x + rng.normal(0, sigma, (draws,5)).cumsum(axis=1)
future_s = np.empty_like(future_x)
last = np.repeat(current_s, draws)
for j in range(5):
    last = .2*future_x[:,j] + .8*last
    future_s[:,j] = last
h = np.arange(1,6)
null_mean = .8**h*current_s + (1-.8**h)*current_x
theory_mse = sigma**2*np.cumsum((1-.8**h)**2)
simulation = pd.DataFrame({'horizon':h, 'analytic_mean':null_mean,
    'MC_mean':future_s.mean(axis=0), 'MC_mean_SE':future_s.std(axis=0,ddof=1)/np.sqrt(draws),
    'analytic_MSE':theory_mse, 'MC_filter_MSE':np.mean((future_s-null_mean)**2, axis=0),
    'MC_persistence_MSE':np.mean((future_s-current_s)**2,axis=0),
    'MC_zero_return_MSE':np.mean((future_x-current_x)**2,axis=0),
    'analytic_zero_return_MSE':sigma**2*h})
simulation.to_csv(OUT/'martingale_control.csv', index=False)
assert np.all(np.abs(simulation.MC_mean-simulation.analytic_mean) < 5*simulation.MC_mean_SE)
display(simulation.round(7))
fig, axes = plt.subplots(1,2, figsize=(11,4.2))
axes[0].plot(h, null_mean, 'o-', label='Analytic filter-aware mean')
axes[0].errorbar(h, future_s.mean(axis=0), yerr=1.96*simulation.MC_mean_SE,
                 fmt='x', capsize=4, label='MC mean ± 1.96 MC SE')
axes[0].axhline(current_s, color='.4', ls='--', label='Smoothed persistence')
axes[0].set(ylabel='Smoothed log level', title='Filter catch-up without return predictability')
axes[1].plot(h, simulation.MC_filter_MSE*1e8, 'o-', label='Filter-aware forecast')
axes[1].plot(h, simulation.MC_persistence_MSE*1e8, 's--', label='Smoothed persistence')
axes[1].set(ylabel='Conditional MSE (log bps squared)', title='Same target, different baselines')
for ax in axes:
    ax.set_xlabel('Forecast horizon (observations)'); ax.legend(fontsize=9)
save_figure(fig, OUT, '02_filter_mechanics_null');
'''),
md('''
The error bars quantify Monte Carlo error in the estimated conditional mean, not
prediction intervals for a future outcome. Filter catch-up alone explains why
smoothed persistence is an inadequate sole benchmark.

### 3. Score both tracks on a random-walk negative control
Parameters are fixed, with no model selection. The smoothed track compares persistence,
smoothed-slope extrapolation, and unchanged/drift price paths propagated through saved filter state.
The return track compares a smoothed-slope candidate against zero and historical raw-drift baselines.
Raw drift is an explicitly labeled benchmark comparator, not a smoothed-only candidate.
'''), code('''
rw = pd.Series(4+np.r_[0., np.random.default_rng(SEED+1).normal(0,.01,1199)].cumsum(),
               index=pd.bdate_range('2020-01-01', periods=1200), name='log_price')
records, rw_s = benchmark(rw, spec, start=250, step=5, filter_id='EMA_0.2')
scores = scorecard(records)
records.to_csv(OUT/'random_walk_forecasts.csv', index=False)
scores.to_csv(OUT/'random_walk_scores.csv', index=False)
for track in ['smoothed','returns']:
    display(scores.loc[scores.track==track, ['horizon','model','n','RMSE','MSE_skill']].round(5))
fig, axes = plt.subplots(1,2, figsize=(11,4.4))
for ax, track in zip(axes, ['smoothed','returns']):
    for model, part in scores[scores.track==track].groupby('model'):
        ax.plot(part.horizon, part.RMSE*1e4, marker='o', label=model)
    ax.set(title=f'{track.title()} track — random-walk control', xlabel='Horizon (observations)',
           ylabel='RMSE (log bps)'); ax.legend(fontsize=8)
save_figure(fig, OUT, '03_separate_scorecards');
'''),
md(r'''
### 4. Inspect session dates and immature labels
An after-close origin forecast is not an executable same-close trade. The project’s
NB9 return convention starts at the next close: $x_{t+1+h}-x_{t+1}$.
That five-session holding target ends six observations after the signal.
We retain this bridge as a distinct label definition; primary milestone targets start at $t$.
'''), code('''
market_x, market_meta = load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',
                                     ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
market_s = pd.Series(smooth(market_x, spec), index=market_x.index)
same = make_targets(market_x, market_s)
delayed = make_targets(market_x, market_s, entry_delay=1)
timing = pd.concat([same.query("track=='returns' and origin_pos==250"),
                    delayed.query("track=='returns' and origin_pos==250")])
display(timing[['origin','entry_delay','horizon','target_start','target_end']])
timing.to_csv(OUT/'target_timing_comparison.csv', index=False)
display(records[records.origin==rw.index[-1]][['track','horizon','model','prediction','status']])
assert records[records.origin==rw.index[-1]].actual.isna().all()
'''),
md('''
## Checks
Verify prefix invariance, poisoned-future immunity, checkpoint equivalence and
nonmutating projection. Future labels are attached only after prediction and are
retained as missing at the current endpoint.
'''), code('''
audit = audit_replay(rw.to_numpy(), spec)
display(pd.DataFrame([audit]))
(OUT/'replay_checks.json').write_text(json.dumps(audit, indent=2))
manifest = write_manifest(OUT, {'filter':filter_configs({'EMA':spec}), 'start':250,
    'step':5, 'horizons':[1,2,3,4,5], 'null_MC_draws':draws,
    'null_assumption':'Gaussian martingale increments; fixed conditional state'},
    {'synthetic':True, 'generator':'known latent signal and separate Gaussian random walk',
     'market_timing_example':market_meta}, seed=SEED)
print('PASS: target maturity, EMA null and causal replay checks.')
'''),
md('''
## Next Steps
TS01 compares EMA variants and adaptive smoothing. TS02 compares endpoint polynomial,
spline, IIR and Kalman estimators. Both retain separate scorecards for the two tracks.
The finite random-walk experiment is a negative control, not a test of stock-market skill.

References: [forecast baselines and rolling-origin validation](https://otexts.com/fpp3/toolbox.html).
The bundled SPY snapshot is a previously examined local project export, ending 2026-10-01;
it supplies actual session dates here, not current market information.
''')])


save("TS01_EMA_and_Adaptive_Filters", [
md('''
# TS01 — EMA families, adaptive smoothing, and lag/noise tradeoffs

## Goal
Compare EMA, DEMA, TEMA and Kaufman adaptive moving average (KAMA), with an
unfiltered reference. Distinguish response mechanics, latent-signal recovery and
forecast skill. No smoother is selected using later market outcomes.
'''), md('''
## Setup
All settings are fixed demonstrations. DEMA and TEMA are signed combinations of
cascaded EMAs. KAMA uses a trailing efficiency ratio and squared adaptive smoothing rate.
Parameters named “window” or “span” are not interchangeable bandwidth measures.
'''), code(SETUP.format(name="TS01_EMA_and_Adaptive_Filters")),
md(r'''
### Filter definitions
For the EMA operator $E$, the cascade outputs are $E(x)$, $E^2(x)$ and $E^3(x)$,
each using the same $\alpha$ and first-observation initialization.
$$\mathrm{DEMA}=2E(x)-E^2(x),\qquad
\mathrm{TEMA}=3E(x)-3E^2(x)+E^3(x).$$
For KAMA over $w$ past differences,
$$ER_t=\frac{|x_t-x_{t-w}|}{\sum_{j=0}^{w-1}|x_{t-j}-x_{t-j-1}|},\qquad
a_t=[ER_t(a_{fast}-a_{slow})+a_{slow}]^2,$$
$$s_t=s_{t-1}+a_t(x_t-s_{t-1}),\qquad a_{span}=2/(span+1).$$
Set $ER_t=0$ when its denominator is zero or the window is incomplete.
Signed EMA combinations can have negative impulse weights; KAMA instead changes
its effective smoothing rate with the observed path. All settings are fixed here.
'''),
md('''
## Steps
### 1. Compare endpoint trajectories against known latent truth
All estimators use the same synthetic observations. Warm-up is suppressed for the
first 60 observations. KAMA is nonlinear and has no single fixed transfer function.
'''), code('''
demo = synthetic_signal(seed=SEED)
specs = {'Unfiltered':FilterSpec(kind='identity'),
         'EMA':FilterSpec(kind='ema',alpha=.2),
         'DEMA':FilterSpec(kind='dema',alpha=.2),
         'TEMA':FilterSpec(kind='tema',alpha=.2),
         'KAMA':FilterSpec(kind='kama',window=10)}
endpoints = pd.DataFrame({name:smooth(demo.observed,spec) for name,spec in specs.items()}, index=demo.index)
endpoints.to_csv(OUT/'causal_endpoints.csv', index_label='Date')
fig, axes = plt.subplots(2,1, figsize=(11,7), sharex=True)
for ax, names in zip(axes, [['EMA','DEMA','TEMA'],['EMA','KAMA']]):
    ax.plot(demo.index, demo.observed, color='.8', lw=.6, label='Noisy observation')
    ax.plot(demo.index, demo.latent, 'k--', lw=1.5, label='Known latent')
    for name in names:
        ax.plot(endpoints.index, endpoints[name], color=COLORS[name], label=name)
    ax.set_ylabel('Log level'); ax.legend(ncol=3, fontsize=9)
axes[0].set_title('Causal endpoint trajectories — fixed settings, unmatched attenuation')
axes[-1].set_xlabel('Synthetic observation date')
save_figure(fig, OUT, '01_endpoint_trajectories');
latent = latent_scores(demo.observed, demo.latent, specs, start=250)
latent.to_csv(OUT/'latent_signal_scores.csv', index=False)
display(latent.round(6))
'''),
md(r'''
### 2. Examine amplitude, delay and overshoot
If $E$ is the EMA operator, DEMA is $2E-E^2$ and TEMA is $3E-3E^2+E^3$.
Low-frequency delay cancellation does not imply uniform zero delay or free denoising.
White-noise gain is $\sum_j w_j^2$ under IID additive observation noise, not an
estimate of removable financial noise. Group delay is frequency-dependent.
'''), code('''
metrics = []
fig, axes = plt.subplots(1,2,figsize=(11,4.4))
for name in ['EMA','DEMA','TEMA']:
    summary, (freq,gain,delay) = response_metrics(specs[name])
    metrics.append({'filter':name, **summary})
    keep = (freq > .001)&(freq <= .15)
    axes[0].plot(freq[keep], np.abs(gain[keep]), color=COLORS[name], label=name)
    axes[1].plot(freq[keep], delay[keep], color=COLORS[name], label=name)
axes[0].set(ylabel='Amplitude gain',title='Signed combinations can amplify a band')
axes[1].set(ylabel='Group delay (sessions)',title='Delay depends on frequency')
for ax in axes:
    ax.set_xlabel('Frequency (cycles / observation)'); ax.legend()
save_figure(fig, OUT, '02_frequency_responses');
metrics = pd.DataFrame(metrics)
metrics.to_csv(OUT/'response_metrics.csv', index=False)
display(metrics.round(5))
'''), code('''
grid = []
for kind in ['ema','dema','tema']:
    for alpha in np.linspace(.05,.5,20):
        summary, (freq,H,delay) = response_metrics(FilterSpec(kind=kind,alpha=float(alpha)))
        passband = (freq >= .005)&(freq <= .025)
        grid.append({'kind':kind,'alpha':alpha,'gain':summary['white_noise_gain'],
                     'passband_mean_delay':np.mean(delay[passband])})
grid = pd.DataFrame(grid)
grid.to_csv(OUT/'lag_noise_grid.csv', index=False)
fig, ax = plt.subplots()
for kind, part in grid.groupby('kind'):
    ax.plot(part.gain,part.passband_mean_delay,'o-',ms=3,color=COLORS[kind.upper()],label=kind.upper())
ax.set(xlabel='IID measurement-noise variance gain',ylabel='Mean delay, 0.005–0.025 cycles/session',
       title='Delay / attenuation tradeoff across alpha values')
ax.legend()
save_figure(fig, OUT, '03_delay_noise_tradeoff');
'''),
md('''
### 3. Test a step change without observation noise
All filters have mature zero-state histories before the step. Overshoot is response
distortion, not forecast anticipation. Adaptive KAMA is shown in the time domain only.
'''), code('''
step = np.r_[np.zeros(250),np.ones(150)]
fig, ax = plt.subplots()
ax.plot(np.arange(len(step))-250, step,'k--',label='True step')
for name in ['EMA','DEMA','TEMA','KAMA']:
    ax.plot(np.arange(len(step))-250,smooth(step,specs[name]),color=COLORS[name],label=name)
ax.set(xlim=(-20,70),xlabel='Observations since step',ylabel='Output / step amplitude',
       title='Step response: speed, overshoot and settling')
ax.legend(ncol=3)
save_figure(fig, OUT, '04_step_response');
'''),
md('''
### 4. Preserve two equal-priority historical scorecards
Use the bundled adjusted SPY snapshot, 2018-01-02 through 2026-10-01. Evaluation
begins at observation 500, advances every five sessions and retains the last origin.
This period has already been explored in this project. There is no tuning here and
these results cannot establish a pristine out-of-sample advantage.
'''), code('''
market_x, metadata = load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',
                                  ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
all_records = []
for name in ['EMA','DEMA','TEMA','KAMA']:
    records, _ = benchmark(market_x, specs[name], start=500, step=5, filter_id=name)
    all_records.append(records)
records = pd.concat(all_records,ignore_index=True)
scores = scorecard(records)
records.to_csv(OUT/'historical_forecasts.csv',index=False)
for track in ['smoothed','returns']:
    scores[scores.track==track].to_csv(OUT/f'{track}_scorecard.csv',index=False)
    display(scores.query('track == @track and horizon == 5')[['filter_id','model','n','RMSE','MSE_skill']].round(5))
fig, axes = plt.subplots(1,2,figsize=(11,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    part = scores[(scores.track==track)&(scores.model=='smooth_slope')]
    for name, group in part.groupby('filter_id'):
        ax.plot(group.horizon,group.MSE_skill,marker='o',color=COLORS[name],label=name)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — smooth-slope candidate',xlabel='Horizon (sessions)',
           ylabel='MSE skill vs track-specific baseline'); ax.legend(fontsize=9)
save_figure(fig, OUT, '05_separate_historical_skill');
'''),
md('''
Skill zero means equality with the stated baseline. The smoothed baseline is
filter-state propagation along an unchanged log-price path; the return baseline is zero.
KAMA propagation is a nonlinear plug-in scenario, not the conditional mean under a
martingale. Each smoothed filter defines a different target: do not rank their raw RMSEs.

## Checks
Replay every filter, preserve state across evaluation boundaries, and verify that
all models within a track/filter/horizon have identical scored origins.
'''), code('''
checks = pd.DataFrame([audit_replay(demo.observed,spec) for spec in specs.values()])
checks.to_csv(OUT/'replay_checks.csv',index=False)
assert scores.groupby(['filter_id','track','horizon']).n.nunique().eq(1).all()
display(checks)
write_manifest(OUT, {'filters':filter_configs(specs), 'start':500,'step':5,
                     'selection':'none; fixed demonstration settings'}, metadata, seed=SEED)
print('PASS: all causal replay and common-origin checks.')
'''), md('''
## Next Steps
TS02 compares estimators at approximately matched IID observation-noise gain.
Neither a smooth appearance nor reduced low-frequency delay establishes predictive skill.
KAMA's adaptive response requires separate simulation-based comparisons.

Reference: [SciPy frequency response](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.freqz.html).
''')])


save("TS02_Endpoint_and_State_Space_Filters", [
md('''
# TS02 — Endpoint polynomial, spline, IIR and Kalman filters

## Goal
Compare causal endpoint estimators, quantify revision hazards, and evaluate both
forecast tracks. Match fixed linear filters on IID observation-noise variance gain
where feasible. Adaptive/time-varying Kalman output is reported separately.
'''), md('''
## Setup
Use fixed parameters, trailing windows, forward IIR state and a filtered local linear
trend. No centered smoother enters a forecast feature or target. The spline uses
fixed lambda in session coordinates; cached endpoint weights exactly implement that
linear trailing spline fit. This is not automatic GCV selection.
'''), code(SETUP.format(name="TS02_Endpoint_and_State_Space_Filters")),
md(r'''
### Estimator definitions
Trailing polynomial regression fits $p(u)=\sum_{j=0}^d\beta_j u^j$ at session
coordinates $u=-(w-1),\ldots,0$ and emits $p(0)=\beta_0$. Endpoint SG uses equal
weights and degree two here. The local-linear comparison uses degree one and
recency weights $1,\ldots,w$. Neither uses observations beyond the current endpoint.

The trailing spline minimizes
$$\sum_{i=1}^w[x_i-f(i)]^2+\lambda\int[f''(u)]^2\,du.$$
It emits the last fitted value, with fixed $\lambda$ in session coordinates.
Butterworth and Bessel use forward-only second-order sections; a saved state
carries the previous filter memory into each new observation. Bessel uses magnitude
normalization at the -3 dB cutoff, rather than a phase-normalized cutoff.

The Kalman model is
$$x_t=[1\;0]z_t+\epsilon_t,\qquad
z_t=\begin{bmatrix}1&1\\0&1\end{bmatrix}z_{t-1}+\eta_t,$$
with $z_t=(level_t,slope_t)$, $Q=\mathrm{diag}(10^{-5},10^{-7})$ and $R=10^{-4}$.
Initialize $z_0=(x_0,0)$ and $P_0=\mathrm{diag}(R,R)$. Only forward filtering is
performed. These fixed log-unit variances are illustrative rather than fitted market estimates.
'''),
md(r'''
## Steps
### 1. Match attenuation using filter mathematics
The reference EMA has $\alpha=0.2$, with IID observation-noise gain $\alpha/(2-\alpha)$.
Choose Butterworth/Bessel cutoffs by deterministic root solving, and trailing polynomial
windows or spline penalties from declared grids. This uses no market outcomes.
Matching one scalar does not match the full frequency response or prove equal forecast difficulty.
'''), code('''
from scipy.optimize import brentq
from scipy.signal import savgol_filter
from scipy.interpolate import make_smoothing_spline
target_gain = .2/1.8
def gain(spec):
    kernel = impulse_response(spec)
    return float(kernel@kernel)
specs = {'EMA':FilterSpec(kind='ema',alpha=.2)}
for name,kind in [('Butterworth','butter'),('Bessel','bessel')]:
    cutoff = brentq(lambda c:gain(FilterSpec(kind=kind,cutoff=c))-target_gain,.002,.3)
    specs[name] = FilterSpec(kind=kind,cutoff=float(cutoff))
for name,kind in [('Endpoint SG','sg'),('Weighted local linear','local_linear')]:
    candidates = [FilterSpec(kind=kind,window=w) for w in range(11,122,2)]
    specs[name] = min(candidates,key=lambda p:abs(gain(p)-target_gain))
spline_candidates = [FilterSpec(kind='spline',window=81,lam=float(lam)) for lam in np.logspace(0,6,31)]
specs['Trailing spline'] = min(spline_candidates,key=lambda p:abs(gain(p)-target_gain))
kalman = FilterSpec(kind='kalman',q_level=1e-5,q_slope=1e-7,r=1e-4)
summary = pd.DataFrame([{'filter':name,**response_metrics(spec)[0],
    'relative_gain_mismatch':gain(spec)/target_gain-1,'window':spec.window,'cutoff':spec.cutoff,'lam':spec.lam}
    for name,spec in specs.items()])
summary.to_csv(OUT/'matched_filter_metrics.csv',index=False)
display(summary.round(5))
'''),
md('''
### 2. Compare common-target denoising and response distortion
Latent RMSE uses the same known latent signal and dates for every estimator. Unlike
cross-filter smoothed-target RMSE, this is a common-target comparison. Parameters are
matched on the declared IID-noise criterion, not optimized on this latent realization.
Kalman settings are illustrative and not attenuation matched.
'''), code('''
demo = synthetic_signal(seed=SEED)
all_specs = {**specs,'Kalman (unmatched)':kalman}
endpoints = pd.DataFrame({name:smooth(demo.observed,spec) for name,spec in all_specs.items()},index=demo.index)
latent = latent_scores(demo.observed,demo.latent,all_specs,start=250)
latent.to_csv(OUT/'common_latent_scores.csv',index=False)
endpoints.to_csv(OUT/'causal_endpoints.csv',index_label='Date')
display(latent.round(6))
fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True)
for ax,names in zip(axes,[['EMA','Butterworth','Bessel'],['Endpoint SG','Weighted local linear','Trailing spline','Kalman (unmatched)']]):
    ax.plot(demo.index,demo.observed,color='.8',lw=.7,label='Observation')
    ax.plot(demo.index,demo.latent,'k--',label='Known latent')
    for name in names:
        ax.plot(endpoints.index,endpoints[name],color=COLORS[name],label=name)
    ax.set_ylabel('Log level'); ax.legend(fontsize=8,ncol=2)
axes[0].set_title('Common observations and latent target; Kalman attenuation unmatched')
axes[-1].set_xlabel('Synthetic observation date')
save_figure(fig, OUT, '01_matched_endpoint_trajectories');
fig, axes = plt.subplots(1,2,figsize=(11,4.5))
step = np.r_[np.zeros(250),np.ones(150)]
for name,spec in specs.items():
    _,(f,H,delay) = response_metrics(spec)
    keep = (f>=.001)&(f<=.12)
    axes[0].plot(f[keep],np.abs(H[keep]),color=COLORS[name],label=name)
    axes[1].plot(np.arange(len(step))-250,smooth(step,spec),color=COLORS[name],label=name)
axes[0].set(xlabel='Frequency (cycles / observation)',ylabel='Amplitude gain',title='Matched noise gain, different responses')
axes[1].plot(np.arange(len(step))-250,step,'k--',label='True step')
axes[1].set(xlim=(-10,80),xlabel='Observations since step',ylabel='Step response',title='Overshoot and settling remain different')
for ax in axes:
    ax.legend(fontsize=8)
save_figure(fig, OUT, '02_response_and_overshoot');
'''),
md('''
### 3. Expose historical revision from retrospective smoothing
Compare outputs at the same historical dates before and after appending observations.
Centered SG and a whole-prefix spline curve revise interior history; archived causal
endpoints do not. A spline evaluated only at each then-current endpoint is a different
time series from the interior curve fitted later.
'''), code('''
cut = 450
raw = demo.observed.to_numpy()
sg_prefix = savgol_filter(raw[:cut],21,2)
sg_later = savgol_filter(raw[:cut+30],21,2)[:cut]
times = np.arange(cut,dtype=float)
spline_prefix = make_smoothing_spline(times,raw[:cut],lam=20)(times)
spline_later = make_smoothing_spline(np.arange(cut+30,dtype=float),raw[:cut+30],lam=20)(times)
causal_prefix = smooth(raw[:cut],specs['Endpoint SG'])
causal_later = smooth(raw[:cut+30],specs['Endpoint SG'])[:cut]
revision = pd.DataFrame({'position':np.arange(cut),
    'centered_SG_revision':sg_later-sg_prefix,
    'whole_curve_spline_revision':spline_later-spline_prefix,
    'archived_endpoint_revision':causal_later-causal_prefix})
revision.to_csv(OUT/'historical_revision.csv',index=False)
assert np.nanmax(np.abs(revision.archived_endpoint_revision)) == 0
fig, ax = plt.subplots()
for name in ['centered_SG_revision','whole_curve_spline_revision','archived_endpoint_revision']:
    ax.plot(revision.position-cut+1,revision[name]*1e4,label=name)
ax.set(xlim=(-40,0),xlabel='Historical observations relative to prefix endpoint',
       ylabel='Revision after appending 30 observations (log bps)',title='Historical revision is observable and testable')
ax.legend(fontsize=9)
save_figure(fig, OUT, '03_historical_revision');
'''),
md('''
### 4. Compare the two forecast tracks without selecting a winner
Same historical SPY snapshot and scheduled origins as TS01. Endpoint settings are
fixed independently of market forecast outcomes. Smoothed track skill is relative
to each filter's own unchanged-path baseline; return track skill uses a common zero-return baseline.
Kalman path propagation remains a scenario for the operational filtered output; a
latent-state forecast is a different target and is outside this milestone.
'''), code('''
market_x,metadata = load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',
                                 ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
frames = []
for name,spec in all_specs.items():
    result,_ = benchmark(market_x,spec,start=500,step=5,filter_id=name)
    frames.append(result)
records = pd.concat(frames,ignore_index=True)
scores = scorecard(records)
records.to_csv(OUT/'historical_forecasts.csv',index=False)
for track in ['smoothed','returns']:
    scores[scores.track==track].to_csv(OUT/f'{track}_scorecard.csv',index=False)
    display(scores.query('track==@track and horizon==5')[['filter_id','model','n','RMSE','MSE_skill']].round(5))
fig,axes = plt.subplots(1,2,figsize=(11,4.6))
for ax,track in zip(axes,['smoothed','returns']):
    for name,part in scores[(scores.track==track)&(scores.model=='smooth_slope')].groupby('filter_id'):
        ax.plot(part.horizon,part.MSE_skill,marker='o',ms=3,color=COLORS[name],label=name)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — smooth-slope candidate',xlabel='Horizon (sessions)',
           ylabel='MSE skill vs track-specific baseline'); ax.legend(fontsize=7)
save_figure(fig, OUT, '04_separate_historical_skill');
'''),
md('''
### 5. Quantify dependence-aware uncertainty for one predeclared comparison
Compare the EMA smooth-slope return candidate with zero at horizon five. Moving-block
resampling uses **scheduled origins**, not individual sessions. Show block-length
sensitivity rather than selecting the most favorable interval. The percentile ranges
are exploratory and conditional on this fixed design, not multiplicity-adjusted claims.
'''), code('''
ema_records = records[records.filter_id=='EMA']
uncertainty = pd.DataFrame([paired_block_difference(ema_records,'returns','smooth_slope','zero',
    horizon=5,block=b,draws=1000,seed=SEED) for b in [5,10,20]])
uncertainty.to_csv(OUT/'paired_block_sensitivity.csv',index=False)
display(uncertainty.round(6))
'''),
md('''
## Checks
Check all seven estimators independently, including checkpoint/resume and poisoned
future data. Every filter/model comparison retains its full scheduled origins,
including immature endpoint forecasts. Full unit tests also verify numerical references.
'''), code('''
checks = pd.DataFrame([{'filter':name,**audit_replay(raw,spec)} for name,spec in all_specs.items()])
checks.to_csv(OUT/'replay_checks.csv',index=False)
assert scores.groupby(['filter_id','track','horizon']).n.nunique().eq(1).all()
assert (scores.groupby(['track','horizon']).n.nunique()==1).all()
display(checks)
write_manifest(OUT, {'filters':filter_configs(all_specs),'start':500,'step':5,
    'matching_target':'IID observation-noise gain alpha/(2-alpha), alpha=.2',
    'matching_uses_market_outcomes':False,'kalman_matched':False},metadata,seed=SEED)
print('PASS: causal replay, matched-origin and immutable-endpoint checks.')
'''), md('''
## Next Steps
Milestone 1 establishes the contracts, filter mechanics and baseline scorecards.
TS03 will cover FFT/decomposition; TS04 will add classical forecasting. The shared
evaluation harness should be extended to nested chronological model selection there.
No filter or forecast candidate has been promoted to deployment status.

References: [endpoint SG](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_coeffs.html),
[smoothing splines](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_smoothing_spline.html),
[forward IIR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html),
[Bessel normalization](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bessel.html).
''')])

print('Generated TS00, TS01 and TS02 notebooks.')
