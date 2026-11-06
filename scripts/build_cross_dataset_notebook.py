"""Generate TS09 only; earlier accepted notebooks remain unchanged."""
from pathlib import Path
from textwrap import dedent
import ast
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s):
 s=dedent(s).strip();ast.parse(s);return nbf.v4.new_code_cell(s)
name='TS09_Cross_Dataset_and_Cross_Asset_Benchmark'
cells=[md(r'''
# TS09 — Cross-dataset and cross-asset forecasting audit

## Goal
Test a fixed representative model menu across six stock histories and two public
climate series, retaining equal-priority smoothed-level and actual-change tracks.
Separate per-dataset chronological selection from aggregate evaluation, preserve
correlated panel structure in resampling, and audit what “independent” can mean.
This is a historical generalization study, not an untouched or prospective test.
'''),md('''
## Setup
Use the existing statsmodels/scikit-learn/CPU PyTorch environment. Inputs, source
metadata, protocol and previous notebooks are bundled; no download or prior notebook
execution is required. `TS09_PROTOCOL.json` was frozen before benchmark execution.
It declares the universe, transforms, budgets, schedules, candidate menu and aggregation.
The complete package is maintained under one stable name for subsequent updates.
'''),code('''
from pathlib import Path
import sys,json
from hashlib import sha256
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src'/'tslab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import asdict
from IPython.display import display,Markdown
from tslab.cross_dataset import CrossSpec,load_cross_inputs,nested_cross,macro_scores,synchronized_panel_intervals
from tslab.reporting import configure_plots,save_figure,write_manifest
from tslab.data import digest
configure_plots();SEED=1731
OUT=ROOT/'outputs'/'TS09_Cross_Dataset_and_Cross_Asset_Benchmark';OUT.mkdir(parents=True,exist_ok=True)
protocol_path=ROOT/'TS09_PROTOCOL.json';protocol=json.loads(protocol_path.read_text())
protocol_digest=digest(protocol_path)
signals,catalogue=load_cross_inputs(ROOT/'inputs/cross_dataset')
assert protocol['input_sha256']=={m['dataset_id']:m['sha256'] for m in catalogue}
stock_ids=[m['dataset_id'] for m in catalogue if m['group']=='stocks']
climate_ids=[m['dataset_id'] for m in catalogue if m['group']=='nonfinancial']
print('Frozen protocol:',protocol_digest)
'''),md(r'''
## Steps
### 1. Inspect provenance, units and exposure before computing scores
Stock inputs use the common observed session index from 2021-01-04 through
2026-10-01: 1,443 rows per asset. The first five caches explicitly adjusted OHLC
by provider Adj Close / Close; the NVDA cache used auto_adjust=True. The producer's
adjustment code and cache byte hashes are preserved. These histories were examined
elsewhere in the project. Adjusted prices are not executable quotes or a point-in-
time corporate-action database; use no pristine test label.

El Niño uses all 732 monthly NOAA temperature observations in the statsmodels
bundle (1950–2010), in **Celsius levels**, not logarithms of Celsius. CO2 uses
monthly means of finite weekly measurements from 1965 onward, in **ppm levels**;
all 444 months have at least one measurement. No interpolation or future filling
is performed. Monthly stamps represent period-end observational availability by
convention; actual historical publication delays and vintages are unavailable.
These climate examples are retrospective signal benchmarks, not operational release-
time forecasts. Stock horizons are at most five sessions; climate horizons are at
most five monthly observations.

The legacy track key `returns` denotes actual future **signal change**. For stocks
it is cumulative log return; for climate it is a Celsius or ppm difference. The
`smoothed` key denotes future causal EMA signal level. Do not pool raw physical-unit
errors or call a temperature difference an investment return.
'''),code('''
quality=[]
for meta in catalogue:
    x=signals[meta['dataset_id']]
    quality.append({'dataset_id':meta['dataset_id'],'group':meta['group'],'rows':len(x),
        'first':x.index[0],'last':x.index[-1],'units':meta['units'],'frequency':meta['frequency'],
        'missing_values':int(x.isna().sum()),'duplicate_dates':int(x.index.duplicated().sum()),
        'input_sha256':meta['sha256'],'pristine_holdout':False})
quality=pd.DataFrame(quality);quality.to_csv(OUT/'data_quality_and_exposure.csv',index=False)
display(quality.drop(columns=['input_sha256']).to_string(index=False))
fig,axes=plt.subplots(2,1,figsize=(11,6))
for identifier in stock_ids:
    x=signals[identifier];axes[0].plot(x.index,np.exp(x-x.iloc[0])*100,label=identifier)
axes[0].set(title='Historical stock inputs — adjusted close indexed to 100',ylabel='Index (first common session = 100)');axes[0].legend(ncol=6,fontsize=8)
# Standardize each physical series for a shape-only view; models use original units.
for identifier in climate_ids:
    x=signals[identifier];axes[1].plot(x.index,(x-x.mean())/x.std(),label=identifier)
axes[1].set(title='Climate inputs — full-series z scores for visualization only',ylabel='Visualization z score',xlabel='Observation date');axes[1].legend()
fig.autofmt_xdate();save_figure(fig,OUT,'01_frozen_dataset_inventory');
'''),md(r'''
### 2. Freeze representative models and nested schedules
All datasets use causal EMA rates 0.2/0.05 and the previous supervised 20 × 2 input.
Ridge(alpha=10), fixed histogram boosting, GRU and NBEATS_demo learn five outputs
from smoothed changes only. AR(5) forecasts smoothed changes natively and uses the
previous restricted OLS bridge to actual signal changes. Thus this is a comparison
of complete procedures, not an isolated architecture experiment.

Neural models retain three fixed seeds, width 12, at most 30 epochs, patience 5,
purged 20% chronological stopping, then same-seed reset/refit on mature examples.
Their predictions are equally averaged; any failed member triggers a whole-family
null fallback. At most 252 mature examples are used. The smaller/earlier climate
fits can have fewer examples; their counts are exported. Parameters are fitted
separately on each dataset; this is **not** a model trained on SPY and transferred
without fitting to another asset.

Climate also includes a period-12 seasonal benchmark: repeat the last observed
12 fast-EMA differences and invert the EMA recursion for future actual changes.
This is not a raw-level seasonal-naive forecast. It receives no future observation.
Nulls are always eligible; drift and slope benchmarks are reported but not selected.

Before each outer block, select separately by dataset and track using equal-horizon
relative MAE at earlier inner origins whose full five-label vectors have matured.
Freeze the choice within that block, while refitting model parameters at each origin.
Stocks: starts 800/1000/1200, six inner origins 20 sessions apart, outer step 10.
El Niño: starts 450/540/630, four inner origins 10 months apart, outer step 10.
CO2: starts 320/360/400, four inner origins 10 months apart, outer step 5.
Only complete five-horizon outer cohorts are scored; latest queries stay unscored.
The delivered model stage is cached with exact protocol, source, output and checkpoint
hashes. Reruns reuse it only when all hashes match; set `RECOMPUTE=True` below to
refit every model. This separates expensive fitting from repeatable chart inspection.
'''),code('''
schedules=[]
for meta in catalogue:
    identifier=meta['dataset_id'];key='stocks' if meta['group']=='stocks' else identifier
    spec=CrossSpec(**{**protocol['schedules'][key],
        'outer_starts':tuple(protocol['schedules'][key]['outer_starts']),
        'seeds':tuple(protocol['schedules'][key]['seeds'])})
    schedules.append({'dataset_id':identifier,**asdict(spec)})
schedule_table=pd.DataFrame(schedules);schedule_table.to_csv(OUT/'declared_schedules.csv',index=False)
display(schedule_table[['dataset_id','outer_starts','outer_step','inner_count','inner_step','max_samples','seasonal_period']])
print('Selection menu: AR5_changes, Ridge_10, HistBoost, GRU, NBEATS_demo; seasonal EMA changes for monthly series; track null always eligible.')
'''),code('''
# False reuses ONLY a completed stage with matching data protocol, code and output hashes.
# Set True for fresh model fitting. The first delivered model stage was computed fresh.
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_cache import load_cache,save_cache
RECOMPUTE=False
combined=None if RECOMPUTE else load_cache(ROOT,OUT,protocol_digest)
if combined is None:
    frames={};runs={}
    for meta in catalogue:
        identifier=meta['dataset_id'];settings=schedule_table.set_index('dataset_id').loc[identifier].to_dict()
        spec=CrossSpec(**settings)
        print('Running',identifier,flush=True)
        run=nested_cross(signals[identifier],spec,checkpoint_dir=OUT/'checkpoints'/identifier,progress=True)
        runs[identifier]=run
        for name,frame in run.items():
            frame=frame.copy();frame['dataset_id']=identifier;frame['dataset_group']=meta['group']
            frame['signal_units']=meta['units'];frame['frequency']=meta['frequency']
            if name in ['forecasts','inner_forecasts']:
                frame['units']=np.where(frame.track=='smoothed',meta['units']+' level',meta['units']+' change')
            frames.setdefault(name,[]).append(frame)
    combined={name:pd.concat(parts,ignore_index=True) for name,parts in frames.items()}
    for name,frame in combined.items():
        frame.to_csv(OUT/('historical_forecasts.csv' if name=='forecasts' else name+'.csv'),index=False)
    save_cache(ROOT,OUT,protocol_digest)
    print('Completed fresh model stage and saved its verified-reuse record.')
else:
    print('Reusing completed model stage: protocol, source, forecasts and checkpoint hashes verified.')
records=combined['forecasts'];scores=combined['scores']
for track in ['smoothed','returns']:scores.query('track==@track').to_csv(OUT/f'{track}_scorecard.csv',index=False)
macro=macro_scores(scores,catalogue);macro.to_csv(OUT/'macro_scorecard.csv',index=False)
panel=synchronized_panel_intervals(records,stock_ids);panel.to_csv(OUT/'synchronized_stock_panel_uncertainty.csv',index=False)
display(combined['selections'][['dataset_id','block','track','chosen_model']].to_string(index=False))
'''),md(r'''
### 3. Compare each dataset before summarizing
MSE skill is $1-MSE_{model}/MSE_{null}$ on identical origins and targets within a
dataset. Positive is better. The null is filter-aware unchanged-price propagation
for the smoothed level and zero change for actual change. Smoothed persistence is
not the principal null. A fixed-family score averages that family's three seeds at
every origin; selected_pipeline is the procedure chosen only from past inner data.
A high family score discovered afterward cannot replace the selected history.

Report results by dataset, horizon and predeclared outer block. Stock targets and
source vintages differ from earlier notebooks' snapshots/cohorts; do not infer a
performance change merely by comparing these numbers with TS07 or TS08 summaries.
'''),code('''
models=['selected_pipeline','AR5_changes','Ridge_10','HistBoost','GRU','NBEATS_demo']
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,track in zip(axes,['smoothed','returns']):
    table=scores[(scores.dataset_id.isin(stock_ids))&(scores.track==track)&(scores.horizon==5)&(scores.model.isin(models))].pivot(index='dataset_id',columns='model',values='MSE_skill').reindex(index=stock_ids,columns=models)
    image=ax.imshow(table.to_numpy(),cmap='RdYlGn',vmin=-.5,vmax=.5,aspect='auto')
    ax.set_xticks(range(len(models)),['Selected','AR5','Ridge','Boost','GRU','NBEATS'],rotation=35,ha='right')
    ax.set_yticks(range(len(stock_ids)),stock_ids);ax.set_title(f'{track.title()} — five-session MSE skill')
    for i in range(len(table)):
        for j in range(len(models)):ax.text(j,i,f'{table.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.grid(False)
    fig.colorbar(image,ax=ax,label='MSE skill (clipped ±0.5)',fraction=.04,pad=.04)
save_figure(fig,OUT,'02_stock_model_comparison');
selected=scores.query('model=="selected_pipeline" and horizon==5')
display(selected[['dataset_id','track','n','MAE','RMSE','MSE_skill']].round(5))
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    for identifier in stock_ids:
        part=scores.query('dataset_id==@identifier and track==@track and model=="selected_pipeline"')
        ax.plot(part.horizon,part.MSE_skill,'o-',label=identifier)
    ax.axhline(0,color='black',ls='--');ax.set(title=f'{track.title()} — selected procedure by asset',xlabel='Horizon (observed sessions)',ylabel='MSE skill vs asset-specific null');ax.legend(fontsize=8,ncol=2)
save_figure(fig,OUT,'03_selected_stock_horizon_profiles');
'''),md('''
### 4. Keep domain aggregation explicit and inspect monthly seasonality
Average within-dataset skill with equal dataset weights; never pool squared Celsius,
ppm and log-price errors. Stocks and nonfinancial datasets have separate aggregates.
The stock basket is a convenience universe from prior work, not a random sample of
assets: it overrepresents related technology names. Six assets on the same 64 dates
are not six independent replications. Some positive/negative dataset counts may
summarize this basket but do not estimate market-wide success rates.

The monthly examples use the same causal smoothed-feature representation, with
an extra seasonal candidate suited to their sampling frequency. A five-month result
is a general library demonstration, not a five-day trading forecast.
'''),code('''
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    for identifier in climate_ids:
        for model,style in [('selected_pipeline','-'),('Seasonal_EMA_changes','--')]:
            part=scores.query('dataset_id==@identifier and track==@track and model==@model')
            ax.plot(part.horizon,part.MSE_skill,'o'+style,label=identifier+' '+('selected' if style=='-' else 'seasonal'))
    label='Smoothed level' if track=='smoothed' else 'Actual change'
    ax.axhline(0,color='black',ls=':');ax.set(title=f'{label} — physical monthly signals',xlabel='Horizon (monthly observations)',ylabel='Within-dataset MSE skill');ax.legend(fontsize=8)
save_figure(fig,OUT,'04_nonfinancial_generalization');
display(macro.query('model=="selected_pipeline" and horizon==5').round(5))
blocks=combined['block_scores'].query('model=="selected_pipeline" and horizon==5')
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    table=blocks[(blocks.track==track)&(blocks.dataset_id.isin(stock_ids))].pivot(index='block',columns='dataset_id',values='MSE_skill')
    for identifier in stock_ids:ax.plot(table.index,table[identifier],'o-',label=identifier)
    ax.axhline(0,color='black',ls='--');ax.set_xticks([0,1,2]);ax.set(title=f'{track.title()} — stock outer-block sensitivity',xlabel='Predeclared chronological block',ylabel='Five-session MSE skill');ax.legend(fontsize=8,ncol=2)
save_figure(fig,OUT,'05_chronological_block_sensitivity');
'''),md(r'''
### 5. Preserve cross-asset dependence in historical uncertainty
Per-dataset moving-block ranges compare selected-minus-null **absolute loss** at
horizon five. Blocks count scheduled origins, not days/months. The synchronized
stock-panel calculation first divides each asset's loss difference by that asset's
full-cohort null MAE, then averages assets equally. The same time blocks are sampled
across all six assets, preserving contemporaneous dependence.

The null denominators remain fixed during this conditional resampling. Ranges are
not MSE-skill confidence intervals, a bootstrap of independent assets or corrections
for prior exploration/model searches. Negative differences favor selection. Treat
block lengths 1/2/4 as sensitivity choices, not a search for a favorable interval.
'''),code('''
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,track in zip(axes,['smoothed','returns']):
    part=panel[panel.track==track]
    ax.vlines(part.block_origins,part.lower95,part.upper95,lw=3)
    ax.plot(part.block_origins,part.mean_normalized_MAE_difference,'o',color='#D55E00')
    ax.axhline(0,color='black',ls='--');ax.set(title=f'{track.title()} — synchronized six-asset panel',xlabel='Time-block length (scheduled origins)',ylabel='Mean normalized MAE difference')
save_figure(fig,OUT,'06_synchronized_panel_uncertainty');
display(panel.round(5))
# Neural seed successes and refit budget are separate from market-loss uncertainty.
neural=combined['neural_audit'];seed_audit=neural[neural.seed.astype(str)!='ensemble']
availability=seed_audit.groupby(['dataset_id','model']).agg(seed_fits=('fit_status','size'),
    successful=('fit_status',lambda s:int(s.eq('ok').sum())),median_refit_epochs=('chosen_epochs','median')).reset_index()
availability.to_csv(OUT/'neural_availability_and_budget.csv',index=False);display(availability)
'''),md('''
## Checks — information timing is necessary, not sufficient for independent evidence
The audit distinguishes computational timing from research independence. Correct
prefixes and nested selection prevent specific leakage mechanisms, but cannot erase
prior inspection, convenience-universe selection or revised historical inputs.
No result here is labeled paper_validated, and no outer winner replaces a frozen
past choice. A prospective specification is bundled; it has not been started or
scheduled and has no collected outcomes.
'''),code('''
assert digest(protocol_path)==protocol_digest
s=combined['selections'];assert (s.inner_last_label<=s.selection_cutoff).all()
for name in ['tabular_audit','neural_audit']:
    a=combined[name];assert (a.max_training_label<=a.origin_pos).all()
a=combined['classical_audit'];assert (a.bridge_last_label_pos<=a.origin_pos).all()
assert scores[scores.dataset_id.isin(stock_ids)].n.eq(64).all()
assert scores.query('dataset_id=="ELNINO"').n.eq(28).all()
assert scores.query('dataset_id=="CO2"').n.eq(24).all()
for identifier in signals:
    latest=signals[identifier].index[-1]
    assert records[(records.dataset_id==identifier)&(records.origin==latest)].status.eq('immature').all()
audit=pd.DataFrame([{'criterion':'Nested selection and label maturity','status':'passed'},
    {'criterion':'Equal within-dataset comparison cohorts','status':'passed'},
    {'criterion':'Input/source and dataset identity hashes','status':'passed'},
    {'criterion':'Cross-asset synchronized resampling','status':'passed'},
    {'criterion':'Pristine independent asset/period holdout','status':'not established'},
    {'criterion':'Point-in-time data vintages / release timing','status':'not established'},
    {'criterion':'Prospective evaluation completed','status':'not completed'},
    {'criterion':'Trading advantage / deployment readiness','status':'not established'}])
audit.to_csv(OUT/'independent_evaluation_audit.csv',index=False);display(audit)
from importlib.metadata import version
manifest=write_manifest(OUT,protocol,{'catalogue':catalogue,'protocol_sha256':protocol_digest},seed=SEED)
manifest['scope']='milestone_6_cross_dataset_historical_generalization_audit'
manifest['versions'].update({p:version(p) for p in ['statsmodels','scikit-learn','torch']})
manifest['checkpoint_sha256']={str(p.relative_to(OUT)):digest(p) for p in sorted((OUT/'checkpoints').rglob('*.pt'))}
manifest['claims']='No pristine independent holdout or prospective evaluation; no established trading advantage'
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
summary='## Observed results'+chr(10)*2+'Six stocks were evaluated on **64 common scored origins** per horizon, plus two monthly physical datasets. '
for row in macro.query('group=="stocks" and model=="selected_pipeline" and horizon==5').itertuples():
    summary+=f'Equal-asset five-session **{row.track}** MSE skill is **{row.mean_MSE_skill:.4f}**, with **{row.positive_datasets}/6** positive asset scores. '
summary+='Per-asset, monthly-dataset, block and synchronized-uncertainty results are reported separately. Historical inputs were previously examined; independence and prospective validation remain unestablished.'
(OUT/'observed_summary.md').write_text(summary)
print('PASS: frozen protocol, mature inner/training labels, matched cohorts and honest independence audit.')
'''),md('''
## Next Steps
TS10 can connect the forecasting library to NB9/NB10 only after explicitly mapping
same-origin targets to executable entry/exit timing. TS09 does not authorize live
trades or imply that historical skill survives transaction costs. The bundled
prospective plan starts no earlier than 2026-11-02 and requires newly issued forecasts
and data-vintage hashes stored before outcomes; it is a specification, not an active
collection service or a completed acceptance test.

Primary references:
[Time-series cross-validation](https://otexts.com/fpp3/toolbox.html#time-series-cross-validation),
[statsmodels El Niño data](https://www.statsmodels.org/stable/datasets/generated/elnino.html),
[statsmodels CO2 data](https://www.statsmodels.org/stable/datasets/generated/co2.html).
Public teaching data and revised stock caches have different provenance limits;
separate their evidence from a future prospective paper-forecast record.
''')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},
    'language_info':{'name':'python','version':'3.12'},'milestone':6,'research_tracks':['smoothed','returns']})
nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb');print('Generated TS09 only.')
