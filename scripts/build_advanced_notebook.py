"""Generate TS07 only; accepted TS00–TS06 are never rewritten."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s):return nbf.v4.new_code_cell(dedent(s).strip())
name='TS07_Advanced_Architecture_Demonstrations'
cells=[md(r'''
# TS07 — Advanced architecture demonstrations and performance interpretation

## Goal
Compare five small architecture demonstrations under the same two-track contracts
as TS06. Understand what the score measures, whether a more complex architecture
helps in this experiment, and how a fixed family differs from a selected procedure.
These are **adaptations for smoothed-input regression**, not reproductions of the
published long-horizon benchmark models. Both research tracks receive equal treatment.
'''),md('''
## Setup
Use the same environment as TS06 (CPU PyTorch >=2.5). No new installation, model
weights, downloads or earlier notebook execution are needed. Preserve the bundled
snapshot and frozen TS04–TS06 procedure references. The snapshot was examined in
earlier experiments; this is exploratory comparison rather than an untouched test.
'''),code('''
from pathlib import Path
import sys,json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src'/'tslab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dataclasses import asdict
from IPython.display import display,Markdown
from tslab.data import load_snapshot
from tslab.supervised import FeatureSpec,SupervisedData,chronological_holdout
from tslab.neural import ForecastNet,NeuralFit,fit_neural,deterministic_cpu
from tslab.advanced import advanced_models,LinearAdapter
from tslab.learned_evaluation import nested_learned
from tslab.learning_reports import export_learning,frozen_comparison
from tslab.provenance4 import write_advanced_manifest
from tslab.reporting import configure_plots,save_figure
configure_plots();deterministic_cpu(1731)
SEED=1731;seeds=(1731,1732,1733)
OUT=ROOT/'outputs'/'TS07_Advanced_Architecture_Demonstrations';OUT.mkdir(parents=True,exist_ok=True)
market_x,metadata=load_snapshot(ROOT/'inputs'/'SPY_adjusted_demo.csv',ROOT/'inputs'/'SPY_adjusted_demo.metadata.json')
features=FeatureSpec(lookback=20,fast_alpha=.2,slow_alpha=.05,warmup=120)
settings=dict(outer_starts=(1000,1400,1800),outer_step=40,inner_count=3,inner_step=40,max_samples=252,min_success=.8)
REF=ROOT/'inputs'/'learning_references'
reference_metadata=json.loads((REF/'metadata.json').read_text())
models=advanced_models();names=[m.name for m in models]
colors=dict(zip(['selected_pipeline']+names,['black','#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']))
'''),md(r'''
## Steps
### 1. Define the architectures and their limits
All models map a **20 × 2** history of fast/slow causal EMA changes to five target
outputs. Track A predicts future fast-EMA log level via residuals around its
unchanged-price catch-up; track B predicts actual cumulative future log returns.
A future smoothed target is easier partly because it averages price innovations;
its score must not be read as an actual-return forecast score.

| Demonstration | Implemented mechanism | Difference from original literature |
|---|---|---|
| DecompositionLinear | Width-5 moving-average trend + remainder, separate shared time projections, channel readout | Inspired by DLinear; regression targets differ from either input channel; no seasonal interpretation claimed |
| LastCenteredLinear | Subtract latest input, temporal linear map, add latest input, channel readout | Inspired by NLinear; operates on EMA changes, not a level being forecast |
| NBEATS | Three generic blocks, each two width-12 hidden layers, residual backcast and summed forecast | Flattened two-channel features; no interpretable polynomial/Fourier bases, original stack depth or benchmark ensemble |
| NHITS | Three residual blocks with time pooling 4/2/1 and forecast knots 1/3/5, linear interpolation | Small multivariate feature adapter, not the original long-horizon implementation |
| PatchTransformer | Four nonoverlapping length-5 joint-channel patches, learned positions, one encoder, width 12, two heads | Patching inspired by PatchTST; **not channel-independent PatchTST**, no RevIN or pretraining |

The decomposition averages only positions inside the already observed input window.
Attention sees all historical patches; no decoder or future token is supplied.
Within-history bidirectional attention is permissible at the forecast origin.
A causal mask would be required for predictions made *inside* that history, which
this demo does not make. All five outputs are direct, with no recursive teacher forcing.
'''),code('''
catalogue=[]
for spec in models:
    network=ForecastNet(spec,features.lookback,2)
    catalogue.append({**asdict(spec),'parameter_count':sum(p.numel() for p in network.parameters()),
                      'input_shape':'20 x 2','output_count':5,'seed_count':len(seeds)})
catalogue=pd.DataFrame(catalogue);catalogue.to_csv(OUT/'architecture_catalogue.csv',index=False)
display(catalogue[['name','parameter_count','max_epochs','patience','seed_count']])
fig,ax=plt.subplots(figsize=(10,4))
ax.barh(names,catalogue.parameter_count,color=[colors[n] for n in names])
ax.set(title='Trainable parameters — compact demonstrations',xlabel='Parameter count')
save_figure(fig,OUT,'01_architecture_sizes');
'''),md(r'''
### 2. Check elementary cases before the stock experiment
If the decomposition's trend and remainder projections are identical, their sum
reduces algebraically to a single linear projection with two biases. The decomposition
alone adds no information. With all model parameters zero, standardized output is zero;
inverse target scaling yields the **training target mean**, not necessarily the
operational baseline. This distinction matters when interpreting standardized losses.

We also report calibration fitting errors on a deterministic planted mapping and
on independent targets. These are **training errors**, not prediction evidence.
They check finite optimization and output shapes without selecting models or tuning
the stock experiment. Limited epochs need not recover even a realizable function.
'''),code('''
torch.manual_seed(SEED)
net=LinearAdapter(20,2,True);net.trend.load_state_dict(net.temporal.state_dict())
probe=torch.randn(4,20,2)
expected=net.readout((net.temporal(probe.transpose(1,2))+net.temporal.bias).transpose(1,2)).squeeze(-1)
torch.testing.assert_close(net(probe),expected,atol=3e-7,rtol=1e-6)
rng=np.random.default_rng(SEED)
Xtoy=rng.normal(size=(100,20,2));toy_origins=np.arange(100,200)
planted=np.stack([.4*Xtoy[:,-1,0]-.2*Xtoy[:,-2,1]+h*.03 for h in range(1,6)],axis=1)
toy_rows=[]
for case,ytoy in [('Planted linear map',planted),('Independent targets',rng.normal(size=(100,5)))]:
    for spec in models:
        fitted,detail=fit_neural(Xtoy,ytoy,toy_origins,spec,seed=SEED)
        predicted=np.array([fitted.predict(seq) for seq in Xtoy])
        mse=np.mean((predicted-ytoy)**2);den=np.mean((ytoy-ytoy.mean(axis=0))**2)
        toy_rows.append({'case':case,'model':spec.name,'training_MSE':mse,
                        'training_MSE_skill_vs_mean':1-mse/den,'chosen_epochs':detail['chosen_epochs']})
toy_table=pd.DataFrame(toy_rows);toy_table.to_csv(OUT/'toy_optimization_checks.csv',index=False)
display(toy_table.round(4))
print('PASS: exact linear-decomposition identity; toy results are in-sample diagnostics only.')
'''),md(r'''
### 3. Hold the training and selection budget fixed
Match TS06: 252 mature examples, 20% chronological stopping tail, purge all training
labels that reach the first stopping feature origin, maximum 30 epochs, patience 5,
AdamW at 0.005, weight decay 0.001, full batch, gradient norm at most 1, CPU and three
fixed seeds. Use stopping-training-only scalers; after epoch choice reset the model
and refit all mature examples and their scalers. No warm starts or best-seed selection.

For 252 examples the split has 196 optimization, 5 purged and 51 validation rows.
Fit all three seeds per architecture/track at every origin. A failed member makes
the whole architecture fall back to its declared baseline and remains visible.
Choose separately by track from three earlier inner origins per block using equal-
horizon relative **MAE**; freeze that choice for the block. Outer scores use 30
origins spaced 40 observed sessions apart; the latest additional origin is unscored.
The last label of every fit is available by the forecast origin.
'''),code('''
dataset=SupervisedData(market_x,features)
X,y,origins=dataset.training(999,'returns',settings['max_samples'])
tr,va=chronological_holdout(origins)
assert (len(tr),len(va))==(196,51)
assert origins[tr[-1]]+5<origins[va[0]] and origins[-1]+5<=999
run=nested_learned(market_x,models,mode='neural',feature_spec=features,seeds=seeds,
    checkpoint_dir=OUT/'checkpoints',**settings,progress=True)
scores,uncertainty=export_learning(run,OUT);records=run['forecasts'];audit=run['fit_audit']
display(run['selections'][['block','track','chosen_model','selection_cutoff']])
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,track in zip(axes,['smoothed','returns']):
    for name in colors:
        part=scores[(scores.track==track)&(scores.model==name)]
        ax.plot(part.horizon,part.MSE_skill,'o-',ms=3,color=colors[name],label=name.replace('_demo',''))
    ax.axhline(0,color='.5',ls='--',lw=1)
    ax.set(title=f'{track.title()} track — 30 scored origins',xlabel='Horizon (sessions)',ylabel='MSE skill vs track baseline')
    ax.legend(fontsize=8)
save_figure(fig,OUT,'02_outer_architecture_skill');
'''),md(r'''
### 4. Read performance without mixing targets, metrics or selection
**MAE** averages absolute errors; **RMSE** emphasizes large misses. Smaller is better.
MSE skill is $1-\mathrm{MSE}_{model}/\mathrm{MSE}_{baseline}$ on identical origins.
Thus +0.20 means 20% lower MSE, equivalent to $\sqrt{0.8}=0.8944$ times baseline
RMSE (10.56% lower RMSE). It does not mean 20% correct predictions or 20% returns.
Negative skill means the baseline did better. Report log errors in log basis points
($10^4$ times log error); these are not exact arithmetic-return basis points.

A **family score** evaluates the same architecture's equal-seed ensemble at every
outer origin. The **selected procedure** can change families or choose the baseline
between blocks using past inner data. Its score can be worse than a family noticed
afterward. That hindsight family is not a valid substitute for the frozen procedure.
Selection admits only these five architecture ensembles plus the track baseline;
filter drift and raw drift are reported diagnostics, not selection candidates. A positive
score against the null does not imply beating these stronger drift benchmarks.
Selection minimizes relative MAE, while training minimizes standardized MSE, so
selection need not maximize reported MSE skill. The tiny inner sample makes choices
noisy. Compare track A only against its filter-aware unchanged-price baseline and
track B only against zero actual return; their skill denominators are different.
'''),code('''
interpret=scores[scores.model.isin(['selected_pipeline','filter_nochange','zero']+names)].copy()
interpret['MAE_log_bps']=interpret.MAE*1e4;interpret['RMSE_log_bps']=interpret.RMSE*1e4
interpret['RMSE_ratio_to_baseline']=np.sqrt(1-interpret.MSE_skill)
interpret.to_csv(OUT/'performance_interpretation.csv',index=False)
for track in ['smoothed','returns']:
    display(interpret.query('track==@track and horizon==5')[['model','n','MAE_log_bps','RMSE_log_bps','MAE_skill','MSE_skill']].round(4))
comparison=frozen_comparison(records,{label:REF/f'{label}.csv' for label in ['TS04_selected','TS05_selected','TS06_selected']},
                            REF/'metadata.json','TS07_selected')
comparison.to_csv(OUT/'matched_prior_pipeline_comparison.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    for label in ['TS04_selected','TS05_selected','TS06_selected','TS07_selected']:
        part=comparison[(comparison.track==track)&(comparison.model==label)]
        ax.plot(part.horizon,part.MSE_skill,'o-',label=label)
    ax.axhline(0,color='.5',ls='--');ax.legend(fontsize=9)
    ax.set(title=f'{track.title()} track — matched dates and targets',xlabel='Horizon (sessions)',ylabel='MSE skill vs common baseline')
save_figure(fig,OUT,'03_matched_procedures');
'''),md('''
### 5. Separate seed sensitivity, fit budgets and historical uncertainty
Seed dots are initialization sensitivity; hollow diamonds are their ensemble scores.
They are not independent market samples or confidence intervals. Epoch counts are
chosen using historical stopping loss. The calibration trace below is predeclared:
patch transformer, first inner origin, actual-return track, first seed.

Moving-block percentile ranges below concern **selected-minus-baseline mean absolute
loss at five sessions**, not MSE skill or the next forecast. Blocks count scheduled
origins, not days. Negative values favor the selected procedure. A range crossing
zero does not resolve a difference under that resampling design. Prior exploration,
multiple architecture searches, asset selection and nonstationarity are not corrected
by these ranges. Forecast-distribution calibration is reserved for TS08.
'''),code('''
seed_scores=pd.read_csv(OUT/'seed_scorecards.csv')
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for ax,track in zip(axes,['smoothed','returns']):
    for j,name in enumerate(names):
        part=seed_scores.query('track==@track and family==@name and horizon==5')
        ax.scatter(j+np.linspace(-.12,.12,len(part)),part.MSE_skill,color=colors[name])
        ensemble=scores.query('track==@track and model==@name and horizon==5').MSE_skill.iloc[0]
        ax.scatter(j,ensemble,marker='D',s=60,facecolors='none',edgecolors='black')
    ax.set_xticks(range(len(names)),[n.replace('_demo','').replace('Linear','\\nLinear').replace('Transformer','\\nTransformer') for n in names],fontsize=8)
    ax.axhline(0,color='.5',ls='--')
    ax.set(title=f'{track.title()} — seeds and ensemble',ylabel='Five-session MSE skill')
save_figure(fig,OUT,'04_seed_sensitivity');
curves=run['learning_curves'];seed_audit=audit[audit.seed.astype(str)!='ensemble']
first=curves.origin_pos.min()
part=curves[(curves.model=='PatchTransformer_demo')&(curves.track=='returns')&(curves.seed==SEED)&(curves.origin_pos==first)]
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
axes[0].plot(part.epoch,part.training_MSE_standardized,label='Optimization')
axes[0].plot(part.epoch,part.validation_MSE_standardized,label='Chronological validation')
axes[0].set(title='Predeclared patch-transformer calibration fit',xlabel='Epoch',ylabel='Mean standardized MSE');axes[0].legend()
axes[1].boxplot([seed_audit.loc[seed_audit.model==n,'chosen_epochs'].dropna() for n in names],
                tick_labels=[n.replace('_demo','').replace('Linear','\\nLinear').replace('Transformer','\\nTransformer') for n in names])
axes[1].tick_params(axis='x',labelsize=8)
axes[1].set(title='Epoch choices — all tracks, seeds and origins',ylabel='Chosen refit epochs')
save_figure(fig,OUT,'05_training_diagnostics');
display(uncertainty.round(7))
fig,axes=plt.subplots(1,2,figsize=(11,4.1))
for ax,track in zip(axes,['smoothed','returns']):
    part=uncertainty[uncertainty.track==track]
    ax.vlines(part.block_origins,part.lower95*1e4,part.upper95*1e4,lw=3)
    ax.plot(part.block_origins,part.mean_MAE_difference*1e4,'o',color='#D55E00')
    ax.axhline(0,color='black',ls='--')
    ax.set(title=f'{track.title()} — selected minus baseline',xlabel='Block length (scheduled origins)',ylabel='Mean MAE difference (log bps)')
save_figure(fig,OUT,'06_paired_uncertainty');
'''),md('''
## Checks
Verify mature labels, purged stopping timing, frozen choices, complete ensemble
averaging and exact inference-checkpoint restoration. Prefix-poisoning tests also
verify that changing observations after a forecast origin cannot change its forecast.
All saved latest forecasts remain unscored until their target sessions are observed.
No outer result chooses architecture sizes, seeds, epochs or smoothing settings.
'''),code('''
assert (audit.max_training_label<=audit.origin_pos).all()
valid=seed_audit[seed_audit.fit_status=='ok']
assert (valid.optimization_last_label<valid.validation_first_origin).all()
assert (valid.last_training_label<=valid.origin_pos).all()
assert (run['selections'].inner_last_label<=run['selections'].selection_cutoff).all()
assert records[records.origin==market_x.index[-1]].status.eq('immature').all()
members=run['seed_forecasts'].groupby(['origin','track','horizon','model']).prediction.mean()
for row in records[(records.model.isin(names))&(records.fit_status=='ok')].itertuples():
    assert abs(row.prediction-members.loc[(row.origin,row.track,row.horizon,row.model)])<1e-12
checks=[]
for path in sorted((OUT/'checkpoints').glob('*.pt')):
    state=torch.load(path,map_location='cpu',weights_only=True)
    restored=NeuralFit.restore(state)
    np.testing.assert_array_equal(restored.predict(state['sequence'].numpy()),state['residual_prediction'].numpy())
    checks.append({'checkpoint':path.name,'restored_prediction_equal':True})
pd.DataFrame(checks).to_csv(OUT/'checkpoint_checks.csv',index=False)
assert len(checks)==len(models)*len(seeds)*2
write_advanced_manifest(OUT,{'features':asdict(features),'models':[asdict(m) for m in models],**settings,
    'seeds':list(seeds),'seed_policy':'equal-weight complete ensemble; never best-seed selection',
    'training':'same deterministic CPU full-batch budget as TS06; purged chronological stopping and reset/refit'},
    {'market':metadata,'references':reference_metadata},seed=SEED)
selected=scores.query("model=='selected_pipeline' and horizon==5")
summary='## Observed results'+chr(10)*2+'Five compact architecture demonstrations were compared with three seeds each. '
for row in selected.itertuples():
    summary+=f'At five sessions, selected **{row.track}** MSE skill is **{row.MSE_skill:.4f}** against **{row.baseline}** across **{row.n}** origins. '
summary+='Compare the family, seed and frozen-procedure results separately. These previously examined single-asset results do not establish a trading advantage.'
(OUT/'observed_summary.md').write_text(summary)
print('PASS: mature labels, purged stopping, exact seed averaging and all checkpoint round trips.')
'''),md('''
## Next Steps
TS08 will address probabilistic forecasts and chronological calibration for both
tracks. Longer histories, wider models and tuning, more assets and prospective
validation require separately specified experiments. This five-session, 20-input-
session comparison does not test the original papers' long-horizon claims.

Primary references:
[N-BEATS](https://arxiv.org/abs/1905.10437),
[N-HiTS](https://arxiv.org/abs/2201.12886),
[DLinear/NLinear](https://arxiv.org/abs/2205.13504),
[PatchTST](https://arxiv.org/abs/2211.14730).
The architecture table specifies what this notebook implements and omits.
''')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','name':'python3','language':'python'},
    'language_info':{'name':'python','version':'3.12'},'milestone':4,'research_tracks':['smoothed','returns']})
nbf.validate(nb);nbf.write(nb,ROOT/'notebooks'/f'{name}.ipynb')
print('Generated TS07 only.')
