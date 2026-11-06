"""Generate TS10 only, preserving canonical NB9/NB10 and earlier TS notebooks."""
from pathlib import Path
from textwrap import dedent
import ast
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s):
 s=dedent(s).strip();ast.parse(s);return nbf.v4.new_code_cell(s)
name='TS10_NB9_NB10_Forecast_and_Trading_Integration'
cells=[md(r'''
# TS10 — NB9/NB10 forecast and trading integration

## Goal
Connect the TS library's two forecast tracks to the exact NB9 next-close target
clock, reproduce the original NB9/NB10 results, and compare a fixed smoothed-input
forecaster under the same research trading rule. Preserve forecast evidence,
execution assumptions, risk allocation and realized wealth as distinct objects.
No original notebook or baseline result is replaced.
'''),md('''
## Setup
The exact NB9 exports, NB10 reference tables/configuration and matching NB3 market/
state inputs are frozen under `inputs/nb_integration`. No network download, previous
notebook execution or live trading connection is required. The matching adjusted
SPY snapshot ends **2026-10-01**. These are previously examined historical inputs;
all fills below are hypothetical adjusted-close bookkeeping, not actual executions.
The fixed experiment is declared in `TS10_PROTOCOL.json` before new scores are viewed.
'''),code('''
from pathlib import Path
import sys,json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/tslab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from tslab.integration import NB9Reference,delayed_samples,summarize_sample
from tslab.handoff import bridge_ledger,bridge_scorecard
from tslab.trading import TradingSpec,decision_plan,simulate_plan,trading_metrics
from tslab.probabilistic import ProbabilitySpec,point_ledger
from tslab.supervised import FeatureSpec
from tslab.reporting import configure_plots,save_figure,write_manifest
from tslab.data import digest
configure_plots();SEED=1731
OUT=ROOT/'outputs'/'TS10_NB9_NB10_Forecast_and_Trading_Integration';OUT.mkdir(parents=True,exist_ok=True)
REF=ROOT/'inputs/nb_integration';protocol_path=ROOT/'TS10_PROTOCOL.json'
protocol=json.loads(protocol_path.read_text());protocol_hash=digest(protocol_path)
assert digest(REF/'reference_manifest.json')==protocol['reference_manifest_sha256']
ref=NB9Reference(REF);price=ref.market.Close.astype(float);log_price=np.log(price)
params=protocol['decision_parameters'];spec=TradingSpec(**{k:v for k,v in params.items() if k in TradingSpec.__dataclass_fields__})
models=protocol['comparison_models']
colors={'analog':'#0072B2','recent':'#E69F00','TS_Ridge_empirical':'#009E73','TS_null_empirical':'#CC79A7','buy_hold':'.4','cash':'black'}
print('Matching market hash:',digest(REF/'nb3/market_ohlcv_snapshot.csv'))
print('Frozen protocol:',protocol_hash,'; snapshot origin:',price.index[-1].date())
'''),md(r'''
## Steps
### 1. Reconcile forecast and execution clocks
The TS native five-step vector describes $R_j=x_{t+j}-x_t$ or smoothed levels
$S_j=s_{t+j}$, $j=1,\ldots,5$. NB9's delayed holding return is
$$Y_{t,h}=x_{t+1+h}-x_{t+1}=R_{h+1}-R_1.$$
The analogous smoothed **change** is $S_{h+1}-S_1$. This is an explicitly derived
smoothed-change target, rather than silently relabeling the native smoothed-level
target. NB10 consumes only the actual-return track.

One-session next-close entry therefore permits holding horizons **1–4** with the
five-step TS vector. NB9's original $h=5$ reaches $t+6$: valid for its original
six-step clock, but outside this library adapter's current budget. Preserve it as
a reference, not a fifth compatible TS holding forecast. Primary holding horizon
is the inherited **h=3**, so entry is $t+1$ and exit is $t+4$.

Differences of marginal quantiles do not describe a difference distribution. Apply
the endpoint subtraction to each **joint sample**, then calculate quantiles and
probabilities. The counterexample below makes that failure visible.
'''),code('''
mapping=pd.DataFrame([{'holding_horizon':h,'signal_offset':0,'entry_offset':1,'exit_offset':1+h,
    'native_steps_required':h+1,'TS_five_step_compatible':h<=4,'status':'compatible' if h<=4 else 'legacy reference only'} for h in range(1,6)])
mapping.to_csv(OUT/'explicit_horizon_mapping.csv',index=False);display(mapping)
sample=np.column_stack([[-2.,-1.,1.,2.],[2.,1.,-1.,-2.],np.zeros((4,3))])
correct=summarize_sample(delayed_samples(sample)[:,0])['q10']
incorrect=summarize_sample(sample[:,1])['q10']-summarize_sample(sample[:,0])['q10']
assert correct==-4 and incorrect==0
print('Counterexample: correct 10th percentile of endpoint difference =',correct,'; subtracting marginal 10th percentiles gives',incorrect)
fig,ax=plt.subplots(figsize=(10,3.2))
for y,label,start,end,color in [(2,'TS native forecast span',0,5,'#0072B2'),(1,'Primary next-close h=3 holding',1,4,'#009E73'),(0,'Legacy NB9 h=5 reference',1,6,'#D55E00')]:
    ax.plot([start,end],[y,y],'o-',lw=4,color=color,label=label)
ax.axvline(5,color='black',ls='--',lw=1);ax.set(xticks=range(7),yticks=[],xlabel='Observed-session offset from signal after close t',title='Explicit timing: a five-step forecast is not a five-session delayed holding')
ax.legend(fontsize=8,loc='center left',bbox_to_anchor=(1.01,.5));save_figure(fig,OUT,'01_forecast_execution_clocks');
'''),md('''
### 2. Replay the frozen NB9 distributions before adding a model
Restore the original five legacy coordinates, frozen calibration metric, k=80,
Theiler exclusion and separated-neighbor weights. All legacy labels mature through
query time and the analog spacing is five sessions. No neighbor-count or support-
threshold reselection is performed. The imported analog uses raw return, volatility
and volume features; the new ridge mean model uses smoothed inputs only. This is a
comparison of complete procedures, not a controlled feature or architecture ablation.

Recompute all three original forecast distributions at all 654 source origins and
horizons 1–5. The smoothed-change analog/recent extension uses the same frozen
neighbors but the TS EMA(alpha=.2) target. The original NB9 did not itself forecast
this smoothed-change target, and its neighbor count was not tuned for that track.
'''),code('''
from tslab.supervised import SupervisedData
dataset=SupervisedData(log_price,FeatureSpec(**protocol['features']))
legacy,neighbor_audit=ref.replay(dataset.smoothed)
legacy.to_csv(OUT/'legacy_replayed_distributions.csv',index=False);neighbor_audit.to_csv(OUT/'legacy_neighbor_audit.csv',index=False)
old=ref.forecasts;new=legacy[legacy.track=='returns']
paired=new.merge(old,on=['date','horizon','model'],suffixes=('_new','_old'),validate='one_to_one')
assert len(paired)==len(old)==9810
checks=[]
for field,source_field in [('mean','mean'),('q10','q10'),('median','median'),('q90','q90'),('p_up','p_up'),('CRPS','crps'),('actual','actual')]:
    a=paired[field+'_new'] if field+'_new' in paired else paired[field]
    b=paired[source_field+'_old'] if source_field+'_old' in paired else paired[source_field]
    difference=np.max(abs(a-b));assert difference<1e-11
    checks.append({'field':field,'rows':len(paired),'max_absolute_difference':difference,'passed':True})
assert (paired.supported_new==paired.supported_old).all()
assert (neighbor_audit.latest_label_pos<=neighbor_audit.origin_pos).all()
nb9_checks=pd.DataFrame(checks);nb9_checks.to_csv(OUT/'legacy_NB9_reproduction_checks.csv',index=False);display(nb9_checks)
'''),md(r'''
### 3. Bridge smoothed-input forecasts using mature issued-error vectors
Fit fixed direct Ridge_10 separately for both native TS tracks using the established
20 × 2 EMA-change inputs and at most 252 examples with all five labels known.
From position 650 onward, issue forecasts daily. At each comparison origin, build
an empirical joint distribution from the last 126 complete **issued forecast-error
vectors**, with $j+5\le t$. Transform its samples to next-close holding outcomes.
Retain a similarly calibrated native null as `TS_null_empirical`; its mean can be
nonzero after residual correction, so it is not the zero-return point forecast.
A separate deterministic zero/filter-aware point control defines point-score skill.

Both new methods use the original NB9 support flag as a **common comparison overlay**;
this is not an intrinsic ridge out-of-distribution detector. Ridge also requires a
successful current fit. There is no future teacher forcing or outer-family tuning.
The error mixture has no asserted stock-data coverage guarantee.
'''),code('''
settings=ProbabilitySpec(forecast_start=650,evaluation_start=1000,max_samples=252,calibration_window=126)
ledger,fit_audit,dataset=point_ledger(log_price,settings,FeatureSpec(**protocol['features']))
ledger.to_csv(OUT/'native_issued_forecast_ledger.csv',index=False);fit_audit.to_csv(OUT/'native_fit_audit.csv',index=False)
support=old.groupby('date').supported.first().copy()
latest_groups,latest_supported,_=ref.neighbors(len(ref.index)-1)
support.loc[price.index[-1]]=latest_supported
origin_dates=pd.DatetimeIndex(sorted(old.date.unique()));origins=price.index.get_indexer(origin_dates)
bridged,calibration_audit,latest_samples=bridge_ledger(ledger,origins,price.index,support,126)
legacy_comparison=legacy[(legacy.model.isin(['analog','recent']))&(legacy.horizon<=4)].copy()
legacy_comparison['status']='scored';legacy_comparison['target_start']=[price.index[t+1] for t in legacy_comparison.origin_pos]
comparison=pd.concat([legacy_comparison,bridged],ignore_index=True)
comparison.to_csv(OUT/'delayed_forecasts.csv',index=False)
calibration_audit.to_csv(OUT/'delayed_calibration_audit.csv',index=False);latest_samples.to_csv(OUT/'latest_delayed_samples.csv',index=False)
scores=bridge_scorecard(comparison);scores.to_csv(OUT/'delayed_forecast_scorecard.csv',index=False)
display(scores.query('horizon==3')[['track','model','n','RMSE','CRPS','MSE_skill']].round(6))
fig,axes=plt.subplots(2,2,figsize=(12,8))
for row,track in enumerate(['smoothed','returns']):
    for model in models:
        part=scores.query('track==@track and model==@model')
        for col,metric in enumerate(['RMSE','CRPS']):axes[row,col].plot(part.horizon,part[metric]*1e4,'o-',color=colors[model],label=model)
    for col,metric in enumerate(['RMSE','CRPS']):
        axes[row,col].set(title=('Smoothed change' if track=='smoothed' else 'Actual return')+' — delayed '+metric,xlabel='Holding horizon after next close',ylabel=metric+' (log bps, lower is better)',xticks=[1,2,3,4]);axes[row,col].legend(fontsize=8)
save_figure(fig,OUT,'02_delayed_forecast_comparison');
'''),md(r'''
### 4. Reproduce NB10's ledger, then apply its rule unchanged
Use only actual-return forecasts. At signal close $t$, enter next close if supported
and mean log return exceeds $\log((1+c)/(1-c))$ plus the inherited five-log-bps buffer.
Allocation is the minimum of 100%, 10% annual-vol target divided by trailing 20-session
annualized volatility, and 1% tail budget divided by the forecast 10th-percentile
loss (floored at $10^{-6}$). The volatility overlay uses raw historical returns,
including $t$; it is risk sizing, not a smoothed-only mean-model predictor.

The fixed primary schedule takes every third source origin, offset zero. It does
not enter when the previous position still overlaps. A scheduled roll sells before
buying at the shared close. For allocated budget $wE$, buy $wE/[P_e(1+c)]$ adjusted-
price units, retain cash, then sell at $P_x(1-c)$. Exact account factor:
$$1+w\left[\frac{P_x}{P_e}\frac{1-c}{1+c}-1\right].$$

This is long-or-cash, fractional adjusted-price bookkeeping, with zero cash interest,
no leverage/shorting, and a lumped assumed proportional cost. Adjusted closes are
not promised market fills. The mean-log threshold is an inherited heuristic, not
an expected-profit guarantee, calibrated significance test or optimal position rule.
'''),code('''
return_forecasts=comparison[(comparison.track=='returns')&(comparison.model.isin(models))&(comparison.status=='scored')].copy()
plans={model:decision_plan(return_forecasts,price,spec,model=model) for model in models}
window=(int(plans['analog'].signal_pos.min()),int(plans['analog'].exit_pos.max()))
results={model:simulate_plan(plan,price,spec,window=window) for model,plan in plans.items()}
buyhold=plans['analog'].iloc[[0]].copy();buyhold['weight']=1.;buyhold['exit_pos']=window[1]
cashplan=plans['analog'].copy();cashplan['weight']=0.
results['buy_hold']=simulate_plan(buyhold,price,spec,window=window);results['cash']=simulate_plan(cashplan,price,spec,window=window)
metrics=pd.DataFrame({m:trading_metrics(*r,spec.initial_capital) for m,r in results.items()}).T.rename_axis('strategy')
metrics.to_csv(OUT/'strategy_metrics.csv')
reproduction=[]
for model in ['analog','recent','buy_hold','cash']:
    imported=pd.read_csv(REF/'nb10'/f'{model}_daily_ledger.csv',index_col=0,parse_dates=True)
    computed=results[model][0]
    assert imported.index.equals(computed.index)
    for col in imported.columns:
        difference=float(np.max(abs(imported[col]-computed[col])))
        np.testing.assert_allclose(imported[col],computed[col],rtol=1e-10,atol=1e-7)
        reproduction.append({'reference':model+'_daily_ledger','field':col,'max_absolute_difference':difference,'passed':True})
original_metrics=pd.read_csv(REF/'nb10/strategy_metrics.csv',index_col=0)
np.testing.assert_allclose(metrics.loc[original_metrics.index,original_metrics.columns],original_metrics,atol=1e-9,rtol=1e-9,equal_nan=True)
pd.DataFrame(reproduction).to_csv(OUT/'legacy_NB10_reproduction_checks.csv',index=False)
for model,(daily,trades) in results.items():
    daily.to_csv(OUT/f'{model}_daily_ledger.csv');trades.to_csv(OUT/f'{model}_trades.csv',index=False)
for model,plan in plans.items():plan.to_csv(OUT/f'{model}_decision_schedule.csv',index=False)
display(metrics[['total_return','max_drawdown','mean_exposure','trades','fees']].round(5))
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
for model,(daily,trades) in results.items():
    axes[0].plot(daily.index,daily.equity/spec.initial_capital,color=colors[model],label=model)
    axes[1].plot(daily.index,daily.drawdown*100,color=colors[model],label=model)
axes[0].set(title='Same-calendar hypothetical net wealth — primary h=3',ylabel='Wealth / initial capital');axes[0].legend(fontsize=8,ncol=2)
axes[1].set(ylabel='Close-marked drawdown (%)',xlabel='Observed session');fig.autofmt_xdate();save_figure(fig,OUT,'03_common_calendar_wealth_and_drawdown');
'''),md('''
### 5. Expose participation and costs without selecting a best schedule
Raw returns differ with exposure and participation; they are not risk-adjusted alpha.
Report zero allocations as scheduled decisions. The assumed cost grid is 0/2/5/10/20
bps **per side**. Keep decisions and allocation fractions fixed while changing the
charged cost; later quantities can differ because account equity differs.

Report every holding horizon 1–4 and every phase offset without choosing the best
retrospectively. Their calendars can differ, so display start/end dates. Reproduce
the inherited analog h=1–5 offset table separately; its h=5 is the original six-step
reference and is not admitted as a five-step TS forecast.
'''),code('''
cost_rows=[]
for model in models+['buy_hold']:
    plan=buyhold if model=='buy_hold' else plans[model]
    for cost in params['cost_grid']:
        daily,trades=simulate_plan(plan,price,spec,cost,window)
        cost_rows.append({'strategy':model,'cost_bps':cost,**trading_metrics(daily,trades,spec.initial_capital)})
costs=pd.DataFrame(cost_rows);costs.to_csv(OUT/'frozen_decision_cost_sensitivity.csv',index=False)
legacy_cost=pd.read_csv(REF/'nb10/cost_sensitivity.csv')
joined=costs[costs.strategy.isin(['analog','recent','buy_hold'])].merge(legacy_cost,on=['strategy','cost_bps'],suffixes=('_new','_old'),validate='one_to_one')
for col in original_metrics.columns:np.testing.assert_allclose(joined[col+'_new'],joined[col+'_old'],atol=1e-8,equal_nan=True)
offsets=[]
for model in models:
    for h in range(1,5):
        for offset in range(h):
            plan=decision_plan(return_forecasts,price,spec,h,model,offset)
            daily,trades=simulate_plan(plan,price,spec)
            offsets.append({'strategy':model,'horizon':h,'offset':offset,'start':daily.index[0],'end':daily.index[-1],**trading_metrics(daily,trades,spec.initial_capital)})
offsets=pd.DataFrame(offsets);offsets.to_csv(OUT/'horizon_offset_sensitivity.csv',index=False)
legacy_offsets=[]
for h in range(1,6):
    for offset in range(h):
        plan=decision_plan(old,price,spec,h,'analog',offset,max_forecast_steps=6)
        daily,trades=simulate_plan(plan,price,spec)
        legacy_offsets.append({'horizon':h,'offset':offset,**trading_metrics(daily,trades,spec.initial_capital)})
legacy_offsets=pd.DataFrame(legacy_offsets);legacy_offsets.to_csv(OUT/'legacy_horizon_offset_reproduction.csv',index=False)
original_offsets=pd.read_csv(REF/'nb10/horizon_offset_sensitivity.csv')
for col in original_metrics.columns:np.testing.assert_allclose(legacy_offsets[col],original_offsets[col],atol=1e-8,equal_nan=True)
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for j,model in enumerate(models):
    part=costs[costs.strategy==model];axes[0].plot(part.cost_bps,part.total_return*100,'o-',color=colors[model],label=model)
    part=offsets[offsets.strategy==model];axes[1].scatter(part.horizon+(j-1.5)*.05,part.total_return*100,color=colors[model],s=24,label=model)
axes[0].set(title='Frozen decision/allocation cost sensitivity',xlabel='One-way assumed cost (bps)',ylabel='Net total return (%)');axes[0].legend(fontsize=8)
axes[1].set(title='All declared holding horizons and offsets',xlabel='Holding horizon (observed sessions)',ylabel='Net total return (%)',xticks=[1,2,3,4]);axes[1].legend(fontsize=8)
save_figure(fig,OUT,'04_cost_and_schedule_sensitivity');
participation=pd.concat([p.reason.value_counts().rename_axis('reason').reset_index(name='scheduled_origins').assign(strategy=m) for m,p in plans.items()])
participation.to_csv(OUT/'participation.csv',index=False);display(participation)
'''),md('''
### 6. Attach dependence-aware uncertainty to the strategy difference
Resample paired daily net returns versus the legacy recent-rule ledger, using
10/20/40-session moving blocks and 2,000 draws. All ledgers share the same calendar.
This is uncertainty for a historical mean daily return difference with fixed rules;
it is not a forecast band, a risk-adjusted alpha test or correction for project-wide
experimentation. Maximum drawdown uses daily closes and misses intraday losses.
'''),code('''
rng=np.random.default_rng(SEED);uncertainty=[]
for model in ['analog','TS_Ridge_empirical','TS_null_empirical']:
    difference=(results[model][0]['return']-results['recent'][0]['return']).iloc[1:].to_numpy()
    for block in protocol['bootstrap']['daily_block_lengths']:
        starts=rng.integers(0,len(difference)-block+1,size=(2000,int(np.ceil(len(difference)/block))))
        ix=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:len(difference)]
        means=difference[ix].mean(axis=1)*1e4
        uncertainty.append({'strategy':model,'baseline':'recent','block_sessions':block,'n_sessions':len(difference),'draws':2000,
            'mean_daily_difference_bps':difference.mean()*1e4,'lower95':np.quantile(means,.025),'upper95':np.quantile(means,.975)})
uncertainty=pd.DataFrame(uncertainty);uncertainty.to_csv(OUT/'paired_strategy_uncertainty.csv',index=False)
fig,ax=plt.subplots(figsize=(10,4.4))
for j,model in enumerate(['analog','TS_Ridge_empirical','TS_null_empirical']):
    part=uncertainty[uncertainty.strategy==model];x=part.block_sessions+(j-1)*1.2
    ax.vlines(x,part.lower95,part.upper95,color=colors[model],lw=2)
    ax.plot(x,part.mean_daily_difference_bps,'o',color=colors[model],label=model)
ax.axhline(0,color='black',ls='--');ax.set(xticks=[10,20,40],xlabel='Paired time-block length (sessions)',ylabel='Mean net daily return difference (bps)',title='Primary h=3: strategy minus legacy recent-rule');ax.margins(y=.15);ax.legend(fontsize=8,ncol=3,loc='upper center')
save_figure(fig,OUT,'05_paired_strategy_uncertainty');display(uncertainty.round(5))
'''),md('''
## Checks and research handoff
Verify exact NB9 forecast replay and NB10 ledger/cost/offset reproduction, mature
native labels and calibration errors, target clocks, no overlapping positions,
self-financing accounting, fees, cash and decisions independent of actual outcomes.
Only actual-return forecasts feed allocations. Smoothed-change scores remain a
separate research track and cannot be interpreted as executable profit forecasts.

The latest snapshot rows below are **research-only handoff records**: no future
fill price or observed exit is supplied, no order is created, and no current-market
recommendation is implied. Cost assumptions, fractional adjusted units and common
legacy support gates remain explicit limitations. No live NB10 input is overwritten.
'''),code('''
assert (fit_audit.max_training_label<=fit_audit.origin_pos).all()
assert (calibration_audit.calibration_last_label_pos<=calibration_audit.origin_pos).all()
assert scores.n.eq(654).all()
assert bridged[bridged.origin_pos==len(price)-1].status.eq('immature').all()
for model,(daily,trades) in results.items():
    if len(trades):
        c=spec.cost_bps/1e4
        np.testing.assert_allclose(trades.net_return,(trades.exit_price/trades.entry_price)*(1-c)/(1+c)-1,atol=1e-12)
        assert np.isclose(trades.entry_fee.sum()+trades.exit_fee.sum(),daily.fee.sum())
    assert np.isclose(np.prod(1+daily['return']),daily.equity.iloc[-1]/spec.initial_capital)
for model,plan in plans.items():
    poisoned=return_forecasts.copy();poisoned['actual']=1e6
    np.testing.assert_array_equal(decision_plan(poisoned,price,spec,model=model).weight,plan.weight)
for model,part in costs.groupby('strategy'):assert (np.diff(part.sort_values('cost_bps').total_return)<=1e-12).all()
latest=bridged[(bridged.origin_pos==len(price)-1)&(bridged.model=='TS_Ridge_empirical')].copy()
latest['entry_delay_sessions']=1;latest['execution_status']='research_only_no_order';latest['source_market_sha256']=protocol['market_sha256']
latest.to_csv(OUT/'research_only_latest_handoff.csv',index=False)
display(latest[['date','track','horizon','forecast_steps','mean','q10','q90','supported','execution_status']].round(6))
assert digest(protocol_path)==protocol_hash
from importlib.metadata import version
manifest=write_manifest(OUT,protocol,{'reference_manifest_sha256':digest(REF/'reference_manifest.json'),'protocol_sha256':protocol_hash},seed=SEED)
manifest['scope']='milestone_7_NB9_NB10_delayed_forecast_and_research_ledger_integration'
manifest['versions']['scikit-learn']=version('scikit-learn')
manifest['claims']='Exact legacy reproduction and historical adjusted-price simulation; no actual orders or independently established trading advantage'
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
summary='## Observed results'+chr(10)*2+f'The common hypothetical trading calendar is **{price.index[window[0]].date()} through {price.index[window[1]].date()}**. '+'The **9,810 legacy NB9 forecast rows** and original NB10 ledgers are reproduced within numerical tolerances. The TS five-step budget supports **1–4 next-close holding sessions**, with **h=3** fixed as primary. '
for model in ['analog','recent','TS_Ridge_empirical','TS_null_empirical','buy_hold']:
    row=metrics.loc[model];summary+=f'**{model}** net total return is **{row.total_return:.2%}** at {spec.cost_bps:g} bps per side (mean exposure {row.mean_exposure:.1%}). '
summary+='Different exposures, historical data reuse and assumed adjusted-close fills prevent reading these differences as established trading alpha. Both delayed forecast tracks have separate scorecards.'
(OUT/'observed_summary.md').write_text(summary)
print('PASS: legacy reproduction, mature clocks, exact ledger identities, cost sensitivity and research-only handoff.')
'''),md('''
## Next Steps
TS11 is the optional foundation-model laboratory; it should retain checkpoint
provenance, training-overlap caveats and explicit compute budgets. Prospective
forecast collection and execution-grade validation remain separate work. This
notebook neither starts a live service nor submits an order.

References: [SEC investor guidance on trade execution](https://www.sec.gov/files/investor/pubs/tradexec.htm),
[Hyndman and Athanasopoulos — forecast combinations and dependence](https://otexts.com/fpp2/practical.html).
The NB9/NB10 reference notebooks and their original output tables are bundled as
immutable provenance inputs, rather than replaced by this integration experiment.
''')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},
 'language_info':{'name':'python','version':'3.12'},'milestone':7,'research_tracks':['smoothed','returns']})
nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb');print('Generated TS10 only.')
