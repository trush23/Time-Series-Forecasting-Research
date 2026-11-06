"""Generate TS03–TS04 only; never rewrite the accepted TS00–TS02 notebooks."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def md(s): return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s): return nbf.v4.new_code_cell(dedent(s).strip())
def save(name,cells):
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},
        'language_info':{'name':'python','version':'3.12'},'milestone':2,
        'research_tracks':['smoothed','returns']})
    nbf.validate(nb)
    nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')


SETUP='''
from pathlib import Path
import sys,json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src'/'tslab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
from IPython.display import display,Markdown
from tslab.data import load_snapshot,synthetic_signal
from tslab.smoothing import FilterSpec,smooth
from tslab.evaluation import scorecard,paired_block_difference
from tslab.reporting import configure_plots,save_figure
from tslab.provenance2 import write_research_manifest
configure_plots()
SEED=1731
OUT=ROOT/'outputs'/'NAME'
OUT.mkdir(parents=True,exist_ok=True)
'''


save('TS03_FFT_and_Bandpass_Decomposition',[
md(r'''
# TS03 — FFT, band-pass decomposition and causal endpoint forecasts

## Goal
Distinguish a retrospectively attractive FFT reconstruction from an estimator that
can actually issue a value at the current endpoint. Then test a deliberately simple
Fourier extrapolator, with separate smoothed-series and actual-return scorecards.
No frequency band is selected by its stock-forecast performance.
'''),md('''
## Setup
The notebook runs independently with the bundled snapshot. Core dependencies remain
NumPy/pandas/SciPy/Matplotlib. The return bridge is shared with TS04 and requires
statsmodels >=0.14.4; no model is downloaded and no prior notebook runs here.
'''),code(SETUP.replace('NAME','TS03_FFT_and_Bandpass_Decomposition')),
md(r'''
## Steps
### 1. Decompose a known multiscale signal
For a finite window, $X_k=\sum_{j=0}^{N-1}x_j e^{-2\pi i jk/N}$ and
$$b_j=\operatorname{IFFT}(M_k X_k).$$
The mask $M_k$ defines the retained band. Frequencies are cycles per observed session.
Low-pass includes DC; band-pass excludes it. A band-passed component alone is not a
price level. We fit an explicit trend, transform its residual, and keep the remainder
so that trend + band + remainder reconstructs the original observations exactly.
All whole-vector decompositions below are retrospective.
'''),code('''
from tslab.spectral import SpectralSpec,SpectralEndpoint,fft_decompose,spectral_replay,harmonic_forecast
from tslab.spectral_evaluation import spectral_benchmark
from scipy.signal import periodogram
n=768
t=np.arange(n)
rng=np.random.default_rng(SEED)
trend=4.5+.0004*t
slow=.04*np.sin(2*np.pi*t/64)
fast=.015*np.cos(2*np.pi*t/12)
observed=trend+slow+fast+rng.normal(0,.008,n)
parts=fft_decompose(observed,low=.01,high=.03,detrend=True)
assert np.max(np.abs(parts['trend']+parts['band']+parts['remainder']-observed))<1e-12
components=pd.DataFrame({'observation':observed,'known_trend':trend,'known_slow':slow,
    'known_fast':fast,'fitted_trend':parts['trend'],'band':parts['band'],'remainder':parts['remainder']})
components.to_csv(OUT/'synthetic_components.csv',index_label='position')
fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
axes[0].plot(t,observed,color='.7',lw=.7,label='Noisy multiscale observation')
axes[0].plot(t,parts['trend'],'k--',label='Fitted global trend')
axes[0].set(ylabel='Log level',title='Retrospective decomposition of a known synthetic signal')
axes[1].plot(t,slow,'k--',label='Known 64-observation component')
axes[1].plot(t,parts['band'],label='FFT band 0.01–0.03 cycles/observation')
axes[1].set_ylabel('Log deviation')
axes[2].plot(t,fast,'k--',label='Known 12-observation component')
axes[2].plot(t,parts['remainder'],lw=.8,label='Remainder, including noise and excluded frequencies')
axes[2].set(xlabel='Synthetic observation position',ylabel='Log deviation')
for ax in axes: ax.legend(fontsize=9)
save_figure(fig,OUT,'01_known_components');
'''),md('''
### 2. Separate power-spectrum estimation from reconstruction
A Hann taper helps spectral leakage in a periodogram but changes the observations.
It is not silently applied to an inverse FFT reconstruction. Finite-window FFT bins
are spaced 1/N cycles per observation; a narrow peak does not establish a persistent
economic cycle. This signal was constructed to contain two known oscillations.
'''),code('''
f,power=periodogram(observed,fs=1,window='hann',detrend='linear',scaling='density')
pd.DataFrame({'frequency_cycles_per_observation':f,'spectral_density':power}).to_csv(OUT/'synthetic_periodogram.csv',index=False)
fig,ax=plt.subplots()
ax.semilogy(f[1:],power[1:],label='Hann periodogram, linear detrending')
for freq,label in [(1/64,'Known period 64'),(1/12,'Known period 12')]:
    ax.axvline(freq,ls='--',label=label)
ax.set(xlim=(0,.15),xlabel='Frequency (cycles / observation)',ylabel='PSD (log-level squared / cycles per observation)',
       title='Power estimation and signal reconstruction are distinct operations')
ax.legend(fontsize=9)
save_figure(fig,OUT,'02_spectrum');
'''),md(r'''
### 3. Archive causal endpoints and measure historical revision
The causal series is $s_t=[D(x_{t-w+1:t})]_{last}$: refit the trailing decomposition
at each date, then keep only the endpoint issued that day. It has no future data,
but still inherits endpoint and periodic-continuation artifacts. A zero-phase
interior curve fitted later is a different estimator.
'''),code('''
spec=SpectralSpec(window=128,low=0.,high=.04,detrend=True)
issued=spectral_replay(observed,spec)
cut=500
prefix=spectral_replay(observed[:cut],spec)
later=spectral_replay(observed[:cut+32],spec)
for key in prefix:
    np.testing.assert_allclose(prefix[key],later[key][:cut],equal_nan=True)
old=fft_decompose(observed[:cut],high=.04)['reconstructed']
new=fft_decompose(observed[:cut+32],high=.04)['reconstructed'][:cut]
revision=pd.DataFrame({'position':np.arange(cut),'retrospective_revision':new-old,
    'archived_endpoint_revision':later['reconstructed'][:cut]-prefix['reconstructed']})
revision.to_csv(OUT/'historical_revision.csv',index=False)
pd.DataFrame(issued).to_csv(OUT/'synthetic_causal_endpoints.csv',index_label='position')
fig,axes=plt.subplots(2,1,figsize=(11,7))
axes[0].plot(t,observed,color='.8',lw=.7,label='Observation')
axes[0].plot(t,fft_decompose(observed,high=.04)['reconstructed'],'--',label='Whole-sample reconstruction (retrospective)')
axes[0].plot(t,issued['reconstructed'],label='Archived trailing endpoints (causal)')
axes[0].set(xlabel='Synthetic observation position',ylabel='Log level',title='Causal endpoints differ from the interior reconstructed curve')
axes[1].plot(revision.position-cut+1,revision.retrospective_revision*1e4,label='Whole-prefix FFT revision')
axes[1].plot(revision.position-cut+1,revision.archived_endpoint_revision*1e4,label='Archived endpoint revision')
axes[1].set(xlim=(-cut,0),xlabel='Observations relative to old endpoint',ylabel='Revision (log bps)',title='Append 32 observations: revision across the old prefix')
for ax in axes: ax.legend(fontsize=9)
save_figure(fig,OUT,'03_causality_and_revision');
'''),md('''
### 4. Stress Fourier extrapolation beyond a stationary cycle
Fourier extrapolation is a forecasting assumption in addition to smoothing. At each
origin, select two eligible FFT bins from the past 128 observations, then jointly
refit trend/sine/cosine coefficients by OLS and extrapolate 1–5 observations.
Compare a stable periodic toy process, a changing-frequency process and a random walk.
These are independent synthetic observed-signal targets, not stock-price forecasts.
No frequency is selected using a future error.

The harmonic fit is **unanchored**: its fitted curve need not meet the last observed
endpoint. A discontinuity there can produce a large one-step error even if the
estimated oscillation looks plausible. This is a property of this declared model,
not proof that all Fourier-based forecasting methods fail.
'''),code('''
toy_rng=np.random.default_rng(SEED+1)
toy_n=640; u=np.arange(toy_n)
toys={'Stable cycle':4+.04*np.cos(2*np.pi*u/32)+toy_rng.normal(0,.004,toy_n),
      'Changing frequency':4+.04*np.sin(2*np.pi*(.015*u+.00006*u**2))+toy_rng.normal(0,.004,toy_n),
      'Random walk':4+toy_rng.normal(0,.01,toy_n).cumsum()}
toy_rows=[]
for name,series in toys.items():
    for origin in range(200,toy_n-5,10):
        forecast,_=harmonic_forecast(series[:origin+1],window=128,harmonics=2)
        for h in range(1,6):
            for method,pred in [('Fourier',forecast[h-1]),('Persistence',series[origin])]:
                toy_rows.append({'process':name,'origin':origin,'horizon':h,'model':method,
                                 'squared_error':(pred-series[origin+h])**2})
toy_records=pd.DataFrame(toy_rows)
toy_scores=toy_records.groupby(['process','horizon','model']).squared_error.mean().pow(.5).rename('RMSE').reset_index()
toy_scores.to_csv(OUT/'toy_forecast_scores.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
for ax,(name,part) in zip(axes,toy_scores.groupby('process')):
    for method,group in part.groupby('model'):
        ax.plot(group.horizon,group.RMSE*1e4,'o-',label=method)
    ax.set(title=name,xlabel='Horizon (observations)');ax.legend(fontsize=8)
axes[0].set_ylabel('RMSE (log bps), common toy target')
save_figure(fig,OUT,'04_fourier_stress_controls');
'''),md(r'''
### 5. Maintain two stock-data research tracks
Use fixed low-pass and trend-plus-band variants of the archived FFT endpoint target.
Each filter defines its own smoothed target; their raw target RMSEs cannot rank the filters.
The stock candidate extrapolates the **smoothed endpoint history**, not raw prices.
The fit audit exports both the fitted endpoint gap and first predicted change, so
the unanchored extrapolator's boundary mismatch is inspectable.

For actual returns, fit only mature historical pairs
$$r_{j+1}=a+b(s_j-s_{j-1})+\varepsilon_{j+1}.$$
Then project those lagged slopes from the Fourier path and sum daily return estimates.
Actual returns remain supervised labels. This is an explicit two-stage return model,
not an assertion that a smoothed change equals an actual return.
'''),code('''
market_x,metadata=load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
specs={'Low-pass':SpectralSpec(window=128,low=0.,high=.04),
       'Trend plus band':SpectralSpec(window=128,low=.01,high=.04)}
records=[];audits=[]
for name,setting in specs.items():
    frame,endpoint,audit=spectral_benchmark(market_x,setting,start=500,step=10)
    frame['filter_label']=name; audit['filter_label']=name
    records.append(frame);audits.append(audit)
records=pd.concat(records,ignore_index=True)
fit_audit=pd.concat(audits,ignore_index=True)
scores=scorecard(records)
records.to_csv(OUT/'historical_forecasts.csv',index=False)
fit_audit.to_csv(OUT/'fit_audit.csv',index=False)
for track in ['smoothed','returns']:
    scores[scores.track==track].to_csv(OUT/f'{track}_scorecard.csv',index=False)
    display(scores.query('track==@track and horizon==5')[['filter_id','model','n','RMSE','MSE_skill']].round(6))
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,track in zip(axes,['smoothed','returns']):
    for label,frame in records.groupby('filter_label'):
        group=scorecard(frame).query("track==@track and model=='Fourier_plus_bridge'")
        ax.plot(group.horizon,group.MSE_skill,'o-',label=label)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — Fourier candidate',xlabel='Horizon (sessions)',ylabel='MSE skill vs track-specific baseline')
    ax.legend(fontsize=9)
save_figure(fig,OUT,'05_separate_stock_scorecards');
'''),md('''
## Checks
Check exact component reconstruction, immutable prefixes, poisoned-future immunity,
checkpoint/resume and nonmutating path propagation. A causal implementation can
still forecast poorly; passing these checks establishes timing correctness, not skill.
'''),code('''
poison=observed.copy();poison[cut:]+=100
poisoned=spectral_replay(poison,spec)
np.testing.assert_allclose(issued['reconstructed'][:cut],poisoned['reconstructed'][:cut],equal_nan=True)
engine=SpectralEndpoint(spec)
for value in observed[:cut]:engine.update(value)
state=engine.checkpoint()
engine.project([observed[cut-1]]*5)
assert state==engine.checkpoint()
restored=SpectralEndpoint.restore(state)
resumed=np.array([restored.update(v)['reconstructed'] for v in observed[cut:]])
np.testing.assert_allclose(resumed,issued['reconstructed'][cut:])
checks={'component_reconstruction':True,'prefix_invariance':True,'future_poison_invariance':True,
        'checkpoint_equivalence':True,'projection_nonmutating':True}
(OUT/'checks.json').write_text(json.dumps(checks,indent=2))
display(pd.DataFrame([checks]))
write_research_manifest(OUT,{'spectral_filters':{k:asdict(v) for k,v in specs.items()},
    'start':500,'step':10,'harmonic_window':128,'harmonics':2,'return_bridge':'OLS next raw return on current smoothed slope',
    'selection':'none; fixed band settings, past-only per-origin amplitude ranking'},metadata,seed=SEED)
maximum=float(np.max(np.abs(revision.retrospective_revision))*1e4)
summary=f"## Observed results\\n\\nAppending 32 observations changes the old whole-prefix FFT curve by up to **{maximum:.2f} log bps** over that historical prefix in this synthetic example; archived causal endpoints have **zero revision**. Both stock tracks retain matched-origin baseline comparisons. The unanchored Fourier extrapolator fits poorly on the stock example; its endpoint gaps are retained in the audit rather than repaired using future outcomes."
(OUT/'observed_summary.md').write_text(summary)
print('PASS: FFT reconstruction, causal state and prefix checks.')
'''),md('''
## Next Steps
TS04 adds fitted classical models and nested chronological model selection. The SPY
snapshot ends 2026-10-01 and has already been examined in this project. These
spectral demonstrations are exploratory, with no untouched holdout or trading claim.

References: [NumPy real FFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html),
[SciPy periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html).
''')])


save('TS04_Classical_Forecasting_and_Selection',[
md(r'''
# TS04 — Classical forecasting with nested chronological selection

## Goal
Compare AR, MA, ARMA, ARIMA, exponential smoothing, damped Holt, Theta and
structural state-space methods at horizons **1–5 observed sessions**. Keep both
research tracks equally explicit, and let a baseline win model selection.
No stock smoother or hyperparameter is tuned on the outer block being evaluated.
'''),md('''
## Setup
Requires statsmodels >=0.14.4 in addition to the milestone 1 environment. The notebook
is self-contained and runs offline from the frozen project snapshot. The default
experiment fits ten candidates on 252-observation training windows, with selection
before three outer blocks. This is a bounded catalogue, not exhaustive AutoARIMA.
'''),code(SETUP.replace('NAME','TS04_Classical_Forecasting_and_Selection')),
md(r'''
## Steps
### 1. Start with a known AR process
For a centered stationary process $d_t=\phi d_{t-1}+\epsilon_t$, its conditional
future mean is $\phi^h d_t$. The demonstration estimates $\phi$ on a known simulation,
before fitting any stock-data candidate. AR/MA/ARMA candidates below model changes
in the smoothed series and integrate their predicted changes back to a level.
'''),code('''
from tslab.classical import default_models,ModelSpec,native_forecast,fit_return_bridge,bridge_forecast
from tslab.prequential import OriginRunner,nested_benchmark
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf
rng=np.random.default_rng(SEED)
phi=.75
d=np.zeros(1200)
for j in range(1,len(d)):d[j]=phi*d[j-1]+rng.normal(0,.01)
estimated=AutoReg(d[:1000],lags=1,trend='c',old_names=False).fit()
h=np.arange(1,6)
estimated_mean=estimated.predict(start=1000,end=1004)
exact_mean=phi**h*d[999]
recovery=pd.DataFrame({'horizon':h,'known_conditional_mean':exact_mean,'estimated_AR_mean':estimated_mean})
recovery.to_csv(OUT/'known_AR_forecast.csv',index=False)
display(pd.DataFrame([{'true_phi':phi,'estimated_phi':estimated.params[1],'estimated_intercept':estimated.params[0]}]))
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
axes[0].plot(d[:120],label='Known AR(1) simulation')
axes[0].set(title='Stationary process with known phi=0.75',xlabel='Observation',ylabel='Simulated value')
axes[1].plot(h,exact_mean,'o-',label='Known conditional mean')
axes[1].plot(h,estimated_mean,'s--',label='Estimated AR mean')
axes[1].set(title='Parameter estimation changes the forecast',xlabel='Horizon (observations)',ylabel='Conditional mean')
for ax in axes:ax.legend(fontsize=9)
save_figure(fig,OUT,'01_known_AR_control');
'''),md(r'''
### 2. State model families and targets precisely
An ARIMA$(p,d,q)$ satisfies $\phi(B)(1-B)^d s_t=c+\theta(B)\epsilon_t$ under
its chosen trend convention. SES is the nonseasonal ETS(A,N,N) point-forecast form;
damped Holt adds an evolving trend with damping $0<\varphi<1$. Theta uses the
nonseasonal theta=2 implementation. Local-level and local-linear-trend models
estimate their structural variances on the training prefix and use forward filtering.

**Track A:** native future smoothed log levels. **Track B:** an explicit two-stage
actual-return model. Fit $r_{j+1}=a+b\Delta s_j+\epsilon_{j+1}$ using only mature
labels, then propagate lagged slopes using the candidate's native smoothed path.
At horizon one all these return candidates have the same currently observed slope,
so differences across native families can appear only at horizons two through five.
The bridge is restricted and may miss information; it is not native ARIMA on returns.
'''),code('''
models=default_models()
model_table=pd.DataFrame([asdict(m) for m in models])
model_table.to_csv(OUT/'model_catalogue.csv',index=False)
display(model_table)
market_x,metadata=load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
filter_spec=FilterSpec(kind='ema',alpha=.2,warmup=60)
settings={'outer_starts':(1000,1400,1800),'outer_step':20,'inner_count':6,'inner_step':25,
          'train_window':252,'min_success':.8}
print('Snapshot:',market_x.index[0].date(),'through',market_x.index[-1].date(),';',len(market_x),'observed sessions')
'''),md('''
### 3. Fit and select using nested chronological windows
Before each outer block, evaluate all ten candidates at six earlier inner origins.
The final five-session inner label must end before the first outer origin. For each
track, choose the candidate with minimum mean relative MAE across the five horizons,
or the track baseline if it scores better. A candidate must fit successfully at
at least 80% of inner origins; otherwise it is ineligible for selection.

Refit model parameters at every scheduled origin using only the most recent 252
mature smoothed observations. Return calibration labels end no later than that origin.
Record all errors/warnings. Failed native fits fall back to the unchanged-path
smoothed baseline or zero return, retaining every scheduled comparison date.
'''),code('''
run=nested_benchmark(market_x,models,filter_spec=filter_spec,**settings)
records=run['forecasts']
scores=scorecard(records)
for name in ['forecasts','inner_forecasts','selections','tuning','fit_audit']:
    filename='historical_forecasts.csv' if name=='forecasts' else name+'.csv'
    run[name].to_csv(OUT/filename,index=False)
run['smoothed'].to_csv(OUT/'causal_smoothed_target.csv',index_label='Date')
for track in ['smoothed','returns']:
    scores[scores.track==track].to_csv(OUT/f'{track}_scorecard.csv',index=False)
display(run['selections'][['block','track','chosen_model','selection_cutoff','outer_first_origin','inner_last_label']])
assert (run['selections'].inner_last_label<=run['selections'].selection_cutoff).all()
assert (run['selections'].selection_cutoff<run['selections'].outer_first_origin).all()
'''),md('''
The outer-origin schedule advances by 20 observed sessions. It is a computationally
bounded forecast experiment, not a daily trading simulation. Final origins whose
future labels have not matured are retained as forecasts and excluded from scores.
The candidate catalogue and selection protocol are fixed for this demonstration;
the underlying SPY period is previously examined, so this is not a pristine holdout.

### 4. Inspect separate scorecards and the selected procedure
The selected pipeline is a procedure that can choose different models by block and
track. Report its losses on outer origins; do not select the best-looking outer model
after seeing the table and call that result validation.
'''),code('''
for track in ['smoothed','returns']:
    display(scores.query('track==@track and horizon==5')[['model','n','MAE','RMSE','MSE_skill']].round(6))
show=['selected_pipeline','AR1_changes','ARIMA111','SES_ETS_ANN','Holt_damped','Theta2','Local_linear_trend']
colors=dict(zip(show,['black','#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9']))
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,track in zip(axes,['smoothed','returns']):
    for name in show:
        group=scores[(scores.track==track)&(scores.model==name)]
        ax.plot(group.horizon,group.MSE_skill,marker='o',ms=3,lw=2 if name=='selected_pipeline' else 1,
                color=colors[name],label=name)
    ax.axhline(0,color='.5',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — outer evaluation',xlabel='Horizon (sessions)',ylabel='MSE skill vs track-specific baseline')
    ax.legend(fontsize=7)
save_figure(fig,OUT,'02_separate_outer_scorecards');
'''),md('''
### 5. Make fit availability and failures visible
These rates include every inner and outer fit once (cached origin/model fits are
not counted twice). Convergence failures invoke the declared fallback and remain in
operational losses. Warnings from successful fits are retained in the audit file.
High forecast accuracy at very low fitting availability must not be hidden.
'''),code('''
availability=run['fit_audit'].groupby(['model','track']).agg(
    n_fits=('fit_status','size'),successful_fraction=('fit_status',lambda x:(x=='ok').mean()),
    mean_runtime_seconds=('runtime_seconds','mean')).reset_index()
availability.to_csv(OUT/'fit_availability.csv',index=False)
display(availability.round(4))
fig,ax=plt.subplots(figsize=(11,5))
part=availability[availability.track=='smoothed'].sort_values('model')
ax.barh(part.model,part.successful_fraction,color='#0072B2')
ax.set(xlim=(0,1.05),xlabel='Successful native-fit fraction, all inner and outer origins',
       title='Fitting availability — failed fits remain in operational scores')
save_figure(fig,OUT,'03_fit_availability');
'''),md('''
### 6. Diagnose a calibration fit without selecting on outer residuals
Inspect standardized residual autocorrelation for a predeclared AR(1)-on-smoothed-
changes model at the first calibration cutoff. ACF is a diagnostic, not proof of
independent errors or predictive adequacy. Native models use smoothed targets;
therefore good residual behavior alone says little about actual-return skill.
'''),code('''
origin=settings['outer_starts'][0]-1
prefix=run['smoothed'].iloc[:origin+1].dropna().to_numpy()
_,diag=native_forecast(prefix,models[0],train_window=settings['train_window'])
residual=diag['residuals']
residual=residual[np.isfinite(residual)]
correlation=acf(residual,nlags=20,fft=False)
pd.DataFrame({'lag':np.arange(len(correlation)),'residual_ACF':correlation}).to_csv(OUT/'calibration_residual_ACF.csv',index=False)
fig,ax=plt.subplots()
ax.stem(np.arange(1,len(correlation)),correlation[1:],basefmt=' ')
ax.axhline(0,color='black',lw=1)
ax.set(xlabel='Lag (observations)',ylabel='Residual autocorrelation',
       title='Predeclared AR(1) calibration diagnostic — no outer residual tuning')
save_figure(fig,OUT,'04_calibration_residual_ACF');
'''),md('''
### 7. Quantify paired uncertainty for the selected procedure
Compare the outer selected pipeline against each track's benchmark at horizon five.
Use moving blocks of scheduled origins; one origin interval is 20 sessions here.
Show three block lengths. These percentile ranges are exploratory conditional
comparisons, not a correction for prior research, model search or nonstationarity.
'''),code('''
uncertainty=[]
for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
    for block in [2,4,8]:
        row=paired_block_difference(records,track,'selected_pipeline',baseline,horizon=5,block=block,draws=2000,seed=SEED)
        uncertainty.append({'track':track,**row})
uncertainty=pd.DataFrame(uncertainty)
uncertainty.to_csv(OUT/'selected_paired_uncertainty.csv',index=False)
display(uncertainty.round(7))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,(track,part) in zip(axes,uncertainty.groupby('track')):
    ax.vlines(part.block_origins,part.lower95*1e4,part.upper95*1e4,color='#0072B2',lw=3)
    ax.plot(part.block_origins,part.mean_MAE_difference*1e4,'o',color='#D55E00')
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — selected minus baseline',xlabel='Block length (scheduled origins)',
           ylabel='Mean MAE difference (log bps)')
save_figure(fig,OUT,'05_paired_uncertainty');
'''),md('''
## Checks
Check historical selection cutoffs, mature bridge labels, identical comparison
origins, frozen current-prefix forecasts under future poisoning, and saved endpoint
forecasts. The full test suite also forces a model failure to verify fallback accounting.
'''),code('''
from tslab.prequential import OriginRunner
probe=int(settings['outer_starts'][0])
original=OriginRunner(market_x.iloc[:probe+6],models[:2],filter_spec).predictions(probe)
poison=market_x.iloc[:probe+6].copy();poison.iloc[probe+1:]+=100
contaminated=OriginRunner(poison,models[:2],filter_spec).predictions(probe)
pd.testing.assert_frame_equal(original,contaminated)
assert (run['fit_audit'].bridge_last_label_pos<=run['fit_audit'].origin_pos).all()
assert scores.groupby(['track','horizon']).n.nunique().eq(1).all()
latest=records[records.origin==market_x.index[-1]]
assert latest.status.eq('immature').all()
latest.to_csv(OUT/'latest_forecasts.csv',index=False)
display(latest[latest.model=='selected_pipeline'][['origin','track','horizon','chosen_model','prediction','fit_status','status']])
checks={'selection_before_outer':True,'bridge_labels_mature':True,'common_outer_origins':True,
        'future_poison_invariant':True,'latest_forecasts_unscored':True}
(OUT/'checks.json').write_text(json.dumps(checks,indent=2))
write_research_manifest(OUT,{'filter':asdict(filter_spec),'models':[asdict(m) for m in models],**settings,
    'selection_objective':'equal-horizon relative MAE, separately per track',
    'fallbacks':{'smoothed':'filter_nochange','returns':'zero'},
    'return_bridge':'OLS next raw return on current smoothed slope'},metadata,seed=SEED)
selected=scores[(scores.model=='selected_pipeline')&(scores.horizon==5)]
summary='## Observed results\\n\\nTen classical candidates were compared using three chronological outer blocks, with separate model choices for each research track. '
for row in selected.itertuples():
    summary+=f"At horizon five, the selected **{row.track}** procedure has MSE skill **{row.MSE_skill:.4f}** against **{row.baseline}** across **{row.n}** scored origins. "
summary+='These are results on previously examined data, not an untouched holdout or an established trading advantage. Fitting failures and baseline fallbacks are retained.'
(OUT/'observed_summary.md').write_text(summary)
print('PASS: selection timing, bridge maturity, prefix forecasts and matched-origin checks.')
'''),md('''
## Next Steps
TS05 will add lagged tabular ML, and TS06 compact neural models, using these same
target and chronological-evaluation contracts. More ARIMA orders, seasonality and
exogenous regressors belong in separate declared grids and appropriate datasets.
Do not impose a five-session seasonal cycle just because horizons run to five.

References: [statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html),
[exponential smoothing](https://www.statsmodels.org/stable/examples/notebooks/generated/exponential_smoothing.html),
[Theta](https://www.statsmodels.org/stable/generated/statsmodels.tsa.forecasting.theta.ThetaModel.html),
[state-space forecasting](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_forecasting.html).
''')])
print('Generated TS03 and TS04; TS00–TS02 were not rewritten.')
