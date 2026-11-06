"""Generate TS05/TS06 only. Existing TS00–TS04 are never rewritten."""
from pathlib import Path
from textwrap import dedent
import argparse
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--only',choices=['both','ts05','ts06'],default='both')
ONLY=parser.parse_args().only


def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s):return nbf.v4.new_code_cell(dedent(s).strip())
def save(name,cells):
    if ONLY!='both' and not name.lower().startswith(ONLY):return
    nb=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},
        'language_info':{'name':'python','version':'3.12'},'milestone':3,
        'research_tracks':['smoothed','returns']})
    nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')


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
from tslab.data import load_snapshot
from tslab.supervised import FeatureSpec,SupervisedData,ema_anchor,chronological_holdout
from tslab.learned_evaluation import LearnedRunner,nested_learned
from tslab.evaluation import scorecard
from tslab.learning_reports import export_learning,frozen_comparison
from tslab.provenance3 import write_learning_manifest
from tslab.reporting import configure_plots,save_figure
configure_plots()
SEED=1731
OUT=ROOT/'outputs'/'NAME';OUT.mkdir(parents=True,exist_ok=True)
market_x,metadata=load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
features=FeatureSpec(lookback=20,fast_alpha=.2,slow_alpha=.05,warmup=120)
settings=dict(outer_starts=(1000,1400,1800),outer_step=40,inner_count=3,inner_step=40,max_samples=252,min_success=.8)
REF=ROOT/'inputs'/'learning_references'
reference_metadata=json.loads((REF/'metadata.json').read_text())
'''


save('TS05_Tabular_Machine_Learning',[
md(r'''
# TS05 — Lagged tabular machine learning and recursive forecasts

## Goal
Compare regularized linear, kernel and tree models with the established baselines.
Use only causal smoothed inputs, retain separate research tracks, and contrast direct
five-horizon prediction with recursive one-step forecasting. No shuffled time split,
future-filled feature or outer-period tuning enters the experiment.
'''),md('''
## Setup
Requires scikit-learn >=1.6 alongside the existing notebook dependencies. The notebook
runs independently from the frozen SPY snapshot; it does not execute previous notebooks.
A bundled TS04 selected-pipeline reference is checksum verified for a matched-date comparison.
'''),code(SETUP.replace('NAME','TS05_Tabular_Machine_Learning')),
md(r'''
## Steps
### 1. Build supervised examples with mature labels
Let $s^f_t$ and $s^s_t$ be EMA levels at rates 0.2 and 0.05. Inputs are the most
recent 20 pairs $(\Delta s^f_j,\Delta s^s_j)$ ending at the feature origin.
They contain no raw price level, actual-return lag, centered smoother or future covariate.

For $q=1-\alpha_f$, the filter-aware unchanged-price catch-up is
$$a_{t,h}=\frac{q}{\alpha_f}(1-q^h)\Delta s^f_t.$$
**Track A** learns the residual $s^f_{t+h}-s^f_t-a_{t,h}$ and adds the anchor back.
**Track B** directly learns the actual log return $x_{t+h}-x_t$ from those smoothed inputs.
All five labels must have matured, so a fit at $t$ can use training origins only through $t-5$.
'''),code('''
from tslab.tabular import tabular_models,TabularSpec,fit_tabular
dataset=SupervisedData(market_x,features)
probe=999
X,y,origins=dataset.training(probe,'returns',settings['max_samples'])
timeline=pd.DataFrame({'training_feature_origin':market_x.index[origins[-8:]],
    'last_label_available':market_x.index[origins[-8:]+5]})
timeline.to_csv(OUT/'training_label_timeline.csv',index=False)
display(timeline)
assert (origins+5<=probe).all()
print('Training shape:',X.shape,'; output shape:',y.shape,'; current query shape:',dataset.features[probe].shape)
fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
for channel,(ax,label) in enumerate(zip(axes,['EMA alpha 0.2 change','EMA alpha 0.05 change'])):
    ax.plot(np.arange(-19,1),dataset.features[probe,:,channel]*1e4,'o-',label=label)
    ax.set_ylabel('Log bps');ax.legend(fontsize=9)
axes[0].set_title('Smoothed-only lag window at a calibration origin')
axes[-1].set_xlabel('Observation offset relative to the current origin')
save_figure(fig,OUT,'01_causal_feature_window');
'''),md('''
### 2. Declare the bounded catalogue
Two ridge penalties are the explicit regularization alternatives. Elastic Net, RBF
SVR, random forest and histogram gradient boosting use fixed documented settings.
Fit feature and target scalers on eligible training examples only. Standardize each
horizon's target separately. Histogram boosting's random early-stopping split is disabled.
The recursive ridge uses the same fully mature training origins as its direct counterpart.
'''),code('''
models=tabular_models()
catalogue=pd.DataFrame([asdict(m) for m in models])
catalogue.to_csv(OUT/'model_catalogue.csv',index=False)
display(catalogue)
print('Forest: 48 trees, depth<=4, min leaf=10, one CPU worker.')
print('Boosting: 60 iterations, depth<=2, learning rate=.05, L2=1, early_stopping=False.')
print('SVR: RBF, C=1, epsilon=.1 on standardized targets; ElasticNet l1_ratio=.5.')
'''),md(r'''
### 3. Check a predictable toy process and a random-walk control
These controls use the same smoothed-input pipeline with one fixed ridge model,
without tuning. In the planted process, actual log returns follow an AR(1) with
$\phi=0.65$; in the negative control they are independent. Results are finite
synthetic demonstrations, not universal performance guarantees or market evidence.
'''),code('''
toy_rng=np.random.default_rng(SEED)
toy_rows=[]
for name,phi in [('Predictable AR returns',.65),('Independent returns',0.)]:
    r=np.zeros(900)
    for j in range(1,900):r[j]=phi*r[j-1]+toy_rng.normal(0,.008)
    toy=pd.Series(4+np.cumsum(r),index=pd.bdate_range('2020-01-01',periods=900))
    runner=LearnedRunner(toy,[TabularSpec('Ridge_10','ridge',10.)],max_samples=252)
    frame=runner.attach_labels(pd.concat([runner.predictions(t) for t in range(650,895,20)],ignore_index=True))
    card=scorecard(frame);card['process']=name;toy_rows.append(card)
toy_scores=pd.concat(toy_rows,ignore_index=True)
toy_scores.to_csv(OUT/'synthetic_control_scores.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,track in zip(axes,['smoothed','returns']):
    for name,part in toy_scores[(toy_scores.track==track)&(toy_scores.model=='Ridge_10')].groupby('process'):
        ax.plot(part.horizon,part.MSE_skill,'o-',label=name)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — synthetic controls',xlabel='Horizon (observations)',ylabel='MSE skill vs track-specific baseline')
    ax.legend(fontsize=8)
save_figure(fig,OUT,'02_synthetic_controls');
'''),md('''
### 4. Run nested chronological selection on the stock snapshot
Use three outer blocks and three earlier inner origins per block. Select separately
by track using equal-horizon relative MAE, with the baseline always eligible. Refit
on at most 252 mature examples at each scheduled origin. Outer origins are 40 sessions
apart: 30 scored origins per horizon, plus the latest unscored forecast.

These budgets are fixed and smaller than the TS04 origin schedule. The snapshot has
already been examined; chronological evaluation does not make it an untouched holdout.
'''),code('''
run=nested_learned(market_x,models,mode='tabular',feature_spec=features,**settings,progress=True)
scores,uncertainty=export_learning(run,OUT)
records=run['forecasts']
display(run['selections'][['block','track','chosen_model','selection_cutoff','inner_last_label']])
for track in ['smoothed','returns']:
    display(scores.query('track==@track and horizon==5')[['model','n','RMSE','MSE_skill']].round(6))
colors=dict(zip(['selected_pipeline']+[m.name for m in models],
    ['black','#0072B2','#56B4E9','#009E73','#CC79A7','#E69F00','#882255','#D55E00']))
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,track in zip(axes,['smoothed','returns']):
    for name in ['selected_pipeline']+[m.name for m in models]:
        part=scores[(scores.track==track)&(scores.model==name)]
        ax.plot(part.horizon,part.MSE_skill,'o-',ms=3,color=colors[name],label=name)
    ax.axhline(0,color='.5',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — tabular outer scores',xlabel='Horizon (sessions)',ylabel='MSE skill vs track-specific baseline')
    ax.legend(fontsize=7)
save_figure(fig,OUT,'03_separate_outer_scorecards');
'''),md(r'''
### 5. Compare direct and recursive strategies without teacher forcing
The direct model predicts five outputs at once. The recursive model fits only its
one-step output and feeds its own projected features forward. EMA increments satisfy
$$\Delta s_{t+1}=q\Delta s_t+\alpha r_{t+1}.$$
For the smoothed residual head, the one-step residual equals $\alpha_f r_{t+1}$.
This identity advances both EMA channels from predictions alone. No realized future
price, return or smoothed endpoint enters the rollout. Both ridge variants use
alpha=10 and the same training origins; each strategy is evaluated without choosing
its best horizon afterward.
'''),code('''
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,track in zip(axes,['smoothed','returns']):
    for name in ['Ridge_10','Ridge_recursive']:
        part=scores[(scores.track==track)&(scores.model==name)]
        ax.plot(part.horizon,part.RMSE*1e4,'o-',color=colors[name],label=name)
    ax.set(title=f'{track.title()} track — same ridge penalty',xlabel='Horizon (sessions)',ylabel='RMSE (log bps)')
    ax.legend(fontsize=9)
save_figure(fig,OUT,'04_direct_vs_recursive');
'''),md('''
### 6. Inspect calibration coefficients, not economic causation
Fit one predeclared ridge model at origin 999 using only mature calibration data.
The coefficients below are in standardized feature/target units. Strongly correlated
lags make individual coefficients unstable; they are descriptive weights, not causal
importance or evidence that an individual lag creates returns.
'''),code('''
X,y,origins=dataset.training(999,'returns',settings['max_samples'])
ridge,diagnostic=fit_tabular(X,y,TabularSpec('Ridge_10','ridge',10.))
coefficients=ridge.model.coef_.reshape(5,features.lookback,2)
rows=[]
for h in range(5):
    for lag in range(features.lookback):
        for channel in range(2):rows.append({'horizon':h+1,'relative_lag':lag-19,'channel':channel,'coefficient':coefficients[h,lag,channel]})
pd.DataFrame(rows).to_csv(OUT/'calibration_ridge_coefficients.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharey=True)
limit=float(np.max(np.abs(coefficients)))
for c,ax in enumerate(axes):
    image=ax.imshow(coefficients[:,:,c],aspect='auto',origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,
                    extent=(-19.5,.5,.5,5.5))
    ax.set(title=['Fast EMA changes','Slow EMA changes'][c],xlabel='Feature lag (observations)')
axes[0].set_ylabel('Forecast horizon (sessions)')
# Reserve the colorbar area before creating it; keep it outside both heatmaps.
fig.subplots_adjust(left=.08,right=.83,bottom=.18,top=.88,wspace=.15)
cax=fig.add_axes([.86,.18,.02,.65])
fig.colorbar(image,cax=cax,label='Standardized ridge coefficient')
fig.savefig(OUT/'05_calibration_coefficients.png',dpi=150,bbox_inches='tight');plt.show()
'''),md('''
### 7. Compare frozen complete pipelines on exactly matched dates
The TS04 reference preserves its original model choices. Restrict it to these outer
origins and verify that targets agree. This compares complete procedures with different
training/selection budgets; it is not an architecture-only experiment. Do not reselect
the earlier procedure from its outer-period performance.
'''),code('''
comparison=frozen_comparison(records,{'TS04_selected':REF/'TS04_selected.csv'},REF/'metadata.json','TS05_selected')
comparison.to_csv(OUT/'matched_prior_pipeline_comparison.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,track in zip(axes,['smoothed','returns']):
    for name in ['TS04_selected','TS05_selected']:
        part=comparison[(comparison.track==track)&(comparison.model==name)]
        ax.plot(part.horizon,part.MSE_skill,'o-',label=name)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — matched outer dates',xlabel='Horizon (sessions)',ylabel='MSE skill vs common baseline');ax.legend(fontsize=9)
save_figure(fig,OUT,'06_matched_pipeline_comparison');
display(uncertainty.round(7))
'''),md('''
## Checks
Verify raw-label maturity, chronological selection, train-only scaling and invariance
to poisoned future data. Current endpoint forecasts remain unscored. Unit tests also
check planted linear mappings, recursive EMA identities and forced failure fallbacks.
'''),code('''
poison=market_x.iloc[:1006].copy();poison.iloc[1001:]+=100
one=LearnedRunner(market_x.iloc[:1006],models[:1]).predictions(1000)
two=LearnedRunner(poison,models[:1]).predictions(1000)
pd.testing.assert_frame_equal(one,two)
assert (run['fit_audit'].max_training_label<=run['fit_audit'].origin_pos).all()
assert (run['selections'].inner_last_label<=run['selections'].selection_cutoff).all()
assert records[records.origin==market_x.index[-1]].status.eq('immature').all()
write_learning_manifest(OUT,{'features':asdict(features),'models':[asdict(m) for m in models],**settings,
    'targets':{'smoothed':'residual above EMA no-change anchor','returns':'actual cumulative log returns'},
    'fit_policy':'independent prefix fits; all five labels mature; train-only feature/target scaling'},
    {'market':metadata,'references':{'TS04_selected':reference_metadata['TS04_selected']}},seed=SEED)
selected=scores.query("model=='selected_pipeline' and horizon==5")
summary='## Observed results'+chr(10)*2+'Seven tabular specifications were tested with separate chronological selection for each track. '
for row in selected.itertuples():summary+=f"At five sessions, selected **{row.track}** MSE skill is **{row.MSE_skill:.4f}** against **{row.baseline}** over **{row.n}** scored origins. "
summary+='These are exploratory results on previously examined data. Direct/recursive comparisons and negative results are retained.'
(OUT/'observed_summary.md').write_text(summary)
print('PASS: mature labels, prefix forecasts, selection timing and latest-origin checks.')
'''),md('''
## Next Steps
TS06 uses the same feature and target definitions for compact neural models, with
chronological early stopping and fixed seed ensembles. Wider hyperparameter searches,
new assets and a prospective period require a separately frozen evaluation protocol.

References: [lagged-feature evaluation](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html),
[ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html).
''')])


save('TS06_Compact_Neural_Forecasting',[
md(r'''
# TS06 — Compact MLP, LSTM, GRU and causal TCN forecasting

## Goal
Evaluate small neural networks with the same smoothed-input and target contracts as
TS05. Measure variability across three fixed seeds, use chronological early stopping,
and compare seed ensembles with baselines and frozen earlier procedures. Complexity
must earn its place through the saved evaluation; no architecture is presumed superior.
'''),md('''
## Setup
Requires PyTorch >=2.5 and scikit-learn >=1.6, alongside the prior dependencies.
All training uses CPU, one PyTorch thread and deterministic algorithms. Each fit starts
from scratch. Exact results may differ across library versions or hardware even with
the same seed. No model download, GPU or previous notebook execution is required.
'''),code(SETUP.replace('NAME','TS06_Compact_Neural_Forecasting')),
md(r'''
## Steps
### 1. Declare architectures, seeds and training budget
Each track has its own five-output network. The input is a 20-by-2 window of causal
EMA increments; the smoothed track predicts residuals above its mechanical anchor,
and the return track directly predicts actual cumulative log returns.

MLP flattens the lag window. LSTM/GRU use one unidirectional recurrent layer and the
last hidden output. The causal TCN uses left padding and dilations 1,2,4,8, giving
receptive field 31 observations (covering the 20-position window with zero left padding).
No right-padding leakage, bidirectional state or future teacher forcing is used.
'''),code('''
import torch
from tslab.neural import neural_models,ForecastNet,fit_neural,NeuralFit
models=neural_models()
seeds=(1731,1732,1733)
catalogue=[]
for model in models:
    network=ForecastNet(model,features.lookback,2)
    catalogue.append({**asdict(model),'parameter_count':sum(p.numel() for p in network.parameters()),
                      'seed_count':len(seeds),'input_shape':'20 x 2','output_count':5})
catalogue=pd.DataFrame(catalogue)
catalogue.to_csv(OUT/'architecture_catalogue.csv',index=False)
display(catalogue)
print('PyTorch:',torch.__version__,'; fixed seeds:',seeds)
print('AdamW; learning rate .005; weight decay .001; gradient norm capped at 1; full-batch training.')
'''),md(r'''
### 2. Purge the chronological stopping split
For each available training pool, reserve its last 20% of feature origins for stopping.
Optimization-training examples must have their full five-session labels end **before**
the first validation feature origin. Overlapping trailing input histories are allowed;
unavailable future labels are not. Stopping-stage scalers use optimization rows only.

Choose the epoch count with minimum chronological validation loss within a maximum
of 30 epochs and patience 5. Then reset the same seed, refit scalers on all mature
examples, and train a fresh network for exactly that many epochs. This uses no future
observation beyond the current forecast origin. The loss is mean MSE across individually
standardized horizon targets; outer model selection still uses relative MAE.
'''),code('''
dataset=SupervisedData(market_x,features)
X,y,origins=dataset.training(999,'returns',settings['max_samples'])
train,val=chronological_holdout(origins)
split_table=pd.DataFrame([{'stage':'optimization','rows':len(train),
    'first_feature':market_x.index[origins[train[0]]],'last_label':market_x.index[origins[train[-1]]+5]},
    {'stage':'stopping validation','rows':len(val),'first_feature':market_x.index[origins[val[0]]],
     'last_label':market_x.index[origins[val[-1]]+5]}])
split_table.to_csv(OUT/'chronological_stopping_split.csv',index=False)
display(split_table)
assert origins[train[-1]]+5<origins[val[0]]
assert origins[val[-1]]+5<=999
'''),md('''
### 3. Evaluate fixed seed ensembles chronologically
At every inner and outer origin, fit all four architectures for both tracks with all
three seeds. Average the three predictions within each architecture; do not select
the best seed. If any seed fails, the architecture uses the predeclared baseline
fallback and records the incomplete ensemble. Individual seed results remain available.

Selection is separate by track and frozen within each outer block. The baseline is
always eligible. The 30 scored outer origins and training budget match TS05.
'''),code('''
run=nested_learned(market_x,models,mode='neural',feature_spec=features,seeds=seeds,
    checkpoint_dir=OUT/'checkpoints',**settings,progress=True)
scores,uncertainty=export_learning(run,OUT)
records=run['forecasts']
display(run['selections'][['block','track','chosen_model','selection_cutoff','inner_last_label']])
for track in ['smoothed','returns']:
    display(scores.query('track==@track and horizon==5')[['model','n','RMSE','MSE_skill']].round(6))
colors={'selected_pipeline':'black','MLP':'#0072B2','LSTM':'#D55E00','GRU':'#009E73','CausalTCN':'#CC79A7'}
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    for name in colors:
        part=scores[(scores.track==track)&(scores.model==name)]
        ax.plot(part.horizon,part.MSE_skill,'o-',ms=3,color=colors[name],label=name)
    ax.axhline(0,color='.5',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — neural ensembles',xlabel='Horizon (sessions)',ylabel='MSE skill vs track-specific baseline');ax.legend(fontsize=8)
save_figure(fig,OUT,'01_separate_outer_scorecards');
'''),md('''
### 4. Inspect calibration learning curves and selected epoch counts
The representative curve is fixed in advance: GRU, return track, first seed, first
inner origin. It is not chosen for appearance or low outer error. Learning curves are
from the purged stopping stage; final refits reset the model and use the chosen epoch
count. All stopping traces are exported for audit.
'''),code('''
curves=run['learning_curves']
first_origin=curves.origin_pos.min()
part=curves[(curves.model=='GRU')&(curves.track=='returns')&(curves.seed==seeds[0])&(curves.origin_pos==first_origin)]
audit=run['fit_audit']
seed_audit=audit[audit.seed.astype(str)!='ensemble'].copy()
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
axes[0].plot(part.epoch,part.training_MSE_standardized,label='Optimization training')
axes[0].plot(part.epoch,part.validation_MSE_standardized,label='Chronological validation')
chosen=seed_audit[(seed_audit.model=='GRU')&(seed_audit.track=='returns')&(seed_audit.seed.astype(str)==str(seeds[0]))&(seed_audit.origin_pos==first_origin)].chosen_epochs.iloc[0]
axes[0].axvline(chosen,color='black',ls='--',label='Chosen refit epoch count')
axes[0].set(title='Predeclared GRU calibration fit',xlabel='Epoch',ylabel='Mean standardized MSE');axes[0].legend(fontsize=8)
names=[m.name for m in models]
epoch_values=[seed_audit.loc[seed_audit.model==name,'chosen_epochs'].dropna() for name in names]
axes[1].boxplot(epoch_values,tick_labels=names,showfliers=True)
axes[1].set(title='Epoch choices across tracks, origins and seeds',ylabel='Chosen refit epochs')
save_figure(fig,OUT,'02_learning_curves_and_stopping');
'''),md('''
### 5. Quantify seed variability without calling it a confidence interval
Show each fixed seed's five-session outer score beside the three-seed ensemble.
These variations reflect training initialization under this fixed data/architecture,
not independent market experiments, forecast intervals or statistical confidence.
Selection uses ensemble predictions only; no favorable seed is promoted.
'''),code('''
seed_scores=pd.read_csv(OUT/'seed_scorecards.csv')
fig,axes=plt.subplots(1,2,figsize=(11,4.2))
for ax,track in zip(axes,['smoothed','returns']):
    for position,name in enumerate(names):
        part=seed_scores[(seed_scores.track==track)&(seed_scores.family==name)&(seed_scores.horizon==5)].sort_values('seed')
        ax.scatter(position+np.linspace(-.12,.12,len(part)),part.MSE_skill,color=colors[name],s=35)
        ensemble=scores[(scores.track==track)&(scores.model==name)&(scores.horizon==5)].MSE_skill.iloc[0]
        ax.scatter(position,ensemble,marker='D',s=65,facecolors='none',edgecolors='black')
    ax.axhline(0,color='.5',ls='--',lw=1);ax.set_xticks(range(len(names)),names)
    ax.set(title=f'{track.title()} track — seed dots, ensemble diamonds',ylabel='Five-session MSE skill')
save_figure(fig,OUT,'03_seed_variability');
'''),md('''
### 6. Compare complete procedures on matched dates
TS04 and TS05 references preserve their delivered block choices. Target values and
dates are checked before comparison. TS04 used different training and selection
budgets; these curves compare procedures, not isolated effects of architecture.
The reference files are frozen: rerunning another notebook does not silently replace them.
'''),code('''
comparison=frozen_comparison(records,{'TS04_selected':REF/'TS04_selected.csv','TS05_selected':REF/'TS05_selected.csv'},
                            REF/'metadata.json','TS06_selected')
comparison.to_csv(OUT/'matched_prior_pipeline_comparison.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.3))
for ax,track in zip(axes,['smoothed','returns']):
    for name in ['TS04_selected','TS05_selected','TS06_selected']:
        part=comparison[(comparison.track==track)&(comparison.model==name)]
        ax.plot(part.horizon,part.MSE_skill,'o-',label=name)
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — matched outer dates',xlabel='Horizon (sessions)',ylabel='MSE skill vs common baseline');ax.legend(fontsize=9)
save_figure(fig,OUT,'04_matched_pipeline_comparison');
'''),md('''
### 7. Keep dependence-aware uncertainty attached to the selected procedure
These moving-block percentile ranges compare selected-minus-baseline absolute loss
at horizon five. Blocks count scheduled origins (40 sessions apart here), not days.
They reflect the stated historical resampling assumptions, not corrections for prior
project exploration or repeated model searches.
'''),code('''
display(uncertainty.round(7))
fig,axes=plt.subplots(1,2,figsize=(11,4.1))
for ax,(track,part) in zip(axes,uncertainty.groupby('track')):
    ax.vlines(part.block_origins,part.lower95*1e4,part.upper95*1e4,lw=3)
    ax.plot(part.block_origins,part.mean_MAE_difference*1e4,'o',color='#D55E00')
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — selected minus baseline',xlabel='Block length (scheduled origins)',ylabel='Mean MAE difference (log bps)')
save_figure(fig,OUT,'05_paired_uncertainty');
'''),md('''
## Checks
Check purged stopping-label timing, mature final-refit labels, frozen selection,
all-seed averaging, and restored checkpoint predictions. The full tests additionally
verify deterministic repeat fits, causal TCN padding and future-poisoned prefix forecasts.
Checkpoints use plain tensor/state dictionaries and load with `weights_only=True`.
'''),code('''
assert (audit.max_training_label<=audit.origin_pos).all()
valid=seed_audit[seed_audit.fit_status=='ok']
assert (valid.optimization_last_label<valid.validation_first_origin).all()
assert (valid.last_training_label<=valid.origin_pos).all()
assert (run['selections'].inner_last_label<=run['selections'].selection_cutoff).all()
assert records[records.origin==market_x.index[-1]].status.eq('immature').all()
checkpoint_checks=[]
for path in sorted((OUT/'checkpoints').glob('*.pt')):
    state=torch.load(path,map_location='cpu',weights_only=True)
    restored=NeuralFit.restore(state)
    prediction=restored.predict(state['sequence'].numpy())
    np.testing.assert_array_equal(prediction,state['residual_prediction'].numpy())
    checkpoint_checks.append({'checkpoint':path.name,'restored_prediction_equal':True})
pd.DataFrame(checkpoint_checks).to_csv(OUT/'checkpoint_checks.csv',index=False)
assert len(checkpoint_checks)==len(models)*len(seeds)*2
write_learning_manifest(OUT,{'features':asdict(features),'models':[asdict(m) for m in models],**settings,
    'seeds':list(seeds),'seed_policy':'equal-weight complete ensemble; never best-seed selection',
    'training':'deterministic CPU, full batch AdamW, purged chronological stopping, reset/refit all mature rows'},
    {'market':metadata,'references':reference_metadata},neural=True,seed=SEED)
selected=scores.query("model=='selected_pipeline' and horizon==5")
summary='## Observed results'+chr(10)*2+'Four compact architectures were evaluated with three fixed seeds each and separate track selection. '
for row in selected.itertuples():summary+=f"At five sessions, selected **{row.track}** MSE skill is **{row.MSE_skill:.4f}** against **{row.baseline}** over **{row.n}** scored origins. "
summary+='Seed variability is reported separately from historical sampling uncertainty. Results remain exploratory; no trading advantage is established.'
(OUT/'observed_summary.md').write_text(summary)
print('PASS: stopping split, mature labels, frozen selection and all checkpoint round trips.')
'''),md('''
## Next Steps
TS07 can demonstrate larger architectures under the same contracts. Wider tuning,
more seeds, additional assets and an untouched prospective period are separate
experiments. A model's complexity or stable training does not establish market skill.

References: [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html),
[GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html),
[reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).
''')])
print('Generated requested TS05/TS06 notebooks; TS00–TS04 untouched.')
