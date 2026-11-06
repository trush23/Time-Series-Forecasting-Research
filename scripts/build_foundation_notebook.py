"""Build the final TS11 notebook without modifying accepted TS00-TS10."""
from pathlib import Path
from textwrap import dedent
import ast
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
def md(s):return nbf.v4.new_markdown_cell(dedent(s).strip())
def code(s):
    s=dedent(s).strip();ast.parse(s);return nbf.v4.new_code_cell(s)
name='TS11_Foundation_Model_Laboratory'
cells=[md(r'''
# TS11 — Foundation-model laboratory

## Goal
Complete the twelve-notebook program with real pretrained forecasting checkpoints,
keeping **future smoothed levels** and **actual future changes from smoothed inputs**
as equal research tracks. Compare Chronos-Bolt tiny and mini at 64/256 observations
with the filter-aware null and fixed Ridge_10 on six stocks and two monthly climate
series. This is a bounded demonstration of one foundation-model family at two sizes,
not a survey of all foundation models or a claim that the newest/largest model wins.
'''),md(r'''
## Setup
**Default: replay the bundled, digest-checked model inference.** All other calculations,
Ridge fits, scoring, calibration and charts are recomputed. This runs in the existing
TS environment without installing Chronos. To rerun the actual neural inference,
install the optional dependencies in `requirements-foundation.txt` in your chosen
Jupyter environment and set `RECOMPUTE=True` below. Local pinned weights are bundled;
neither mode downloads anything. The timing chart always identifies the original
fresh-inference run. See `FOUNDATION_MODEL_GUIDE.md` for exact reproduction steps.

The protocol was frozen before TS11 forecasts/scores. Model inputs are centered,
trailing **EMA levels**, alpha .2; no raw price predictor or full-sample smoother.
Both contexts are reported without tuning. Context 256 and horizon 5 are declared
primary. Forty calibration origins precede forty evaluation origins, then one latest
unscored origin. Stocks use every fifth observed session; climate series every month.
Stock horizons are 1–5 sessions; climate horizons are 1–5 **months**.

### Key assumptions
Existing histories and adjusted-price vintages were previously examined. Checkpoint
training end dates and exact corpus membership are not established for these data.
The source vendor's general zero-shot benchmark claims do not establish zero-shot
independence here. The two checkpoints were pretrained; no project fine-tuning occurs.
This notebook neither changes TS10 trading rules nor issues orders.
'''),code('''
from pathlib import Path
import sys,json
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'src/tslab').is_dir())
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
from tslab.cross_dataset import load_cross_inputs
from tslab.supervised import SupervisedData
from tslab.foundation import (foundation_quantiles,replay,score_points,residual_intervals,
    paired_ranges,martingale_control,invert_ema_path,centered_context,schedule)
from tslab.reporting import configure_plots,save_figure,write_manifest
from tslab.data import digest
configure_plots();RECOMPUTE=False
OUT=ROOT/'outputs'/'TS11_Foundation_Model_Laboratory';OUT.mkdir(parents=True,exist_ok=True)
protocol=json.loads((ROOT/'TS11_PROTOCOL.json').read_text())
signals,catalogue=load_cross_inputs(ROOT/'inputs/cross_dataset')
checkpoints=json.loads((ROOT/'inputs/foundation_models/checkpoint_manifest.json').read_text())
assert digest(ROOT/'inputs/foundation_models/checkpoint_manifest.json')==protocol['checkpoint_manifest_sha256']
assert {c['model_id']:c['revision'] for c in checkpoints}=={c['id']:c['revision'] for c in protocol['models']}
assert {m['dataset_id']:m['sha256'] for m in catalogue}==protocol['input_sha256']
labels={'filter_null':'Null','Ridge_10':'Ridge','chronos-bolt-tiny_L64':'Tiny 64',
    'chronos-bolt-tiny_L256':'Tiny 256','chronos-bolt-mini_L64':'Mini 64','chronos-bolt-mini_L256':'Mini 256'}
order=list(labels);colors=dict(zip(order,['#555555','#009E73','#56B4E9','#0072B2','#E69F00','#D55E00']))
print('Protocol:',digest(ROOT/'TS11_PROTOCOL.json'))
print('Mode:', 'fresh offline CPU inference' if RECOMPUTE else 'audited inference replay; fresh Ridge/scoring/calibration')
'''),md(r'''
## Steps
### 1. Start with an identity and a counterexample
For EMA $s_t=\alpha x_t+(1-\alpha)s_{t-1}$, a deterministic future EMA path implies
$$\widehat{x}_{t+h}={\widehat{s}_{t+h}-(1-\alpha)\widehat{s}_{t+h-1}\over\alpha},
\qquad \widehat{r}_{t,h}=\widehat{x}_{t+h}-{s_t-(1-\alpha)s_{t-1}\over\alpha}.$$
Set $\widehat{s}_t=s_t$. This uses only the smoothed history, including the last two
values. EMA is algebraically invertible, so smoothed-only inputs do **not** imply
information reduction. Low lag and denoising do not create return predictability.

The foundation return forecast is the inverse of its **marginal median level path**,
a deterministic point proxy. It is not generally a conditional return mean/median.
At alpha .2, the inverse has coefficients 5 and -4 and can amplify forecast noise.
Marginal quantiles cannot be differenced into a distribution; the counterexample
below has identical endpoint marginal distributions but opposite dependence.
We therefore apply the same earlier-error interval calibration to both tracks and
all six procedures. No fabricated native return interval is reported.
'''),code('''
a=.2;q=1-a;x0=10.;s0=9.;sp=(s0-a*x0)/q
hs=np.arange(1,6);null_path=s0+(x0-s0)*(1-q**hs)
np.testing.assert_allclose(invert_ema_path(null_path,s0,sp),0,atol=2e-14)
u=np.array([-2.,-1.,1.,2.]);same=u-u;opposite=-u-u
counter=pd.DataFrame({'coupling':['same endpoints','opposite endpoints'],
    'endpoint1_q10':[np.quantile(u,.1)]*2,'endpoint2_q10':[np.quantile(u,.1)]*2,
    'difference_q10':[np.quantile(same,.1),np.quantile(opposite,.1)]})
counter.to_csv(OUT/'marginal_quantile_counterexample.csv',index=False);display(counter)
catalog=pd.DataFrame(catalogue)[['dataset_id','group','rows','first','last','frequency','units']]
catalog.to_csv(OUT/'evaluation_catalogue.csv',index=False);display(catalog)
provenance=pd.DataFrame([{'model':c['model_id'],'revision':c['revision'],'training_end':'unknown',
    'overlap_with_each_evaluation_dataset':'unknown','license':c['license']} for c in checkpoints])
provenance.to_csv(OUT/'checkpoint_provenance.csv',index=False);display(provenance)
'''),md('''
### 2. Run or replay the real checkpoints; keep target maturity explicit
Each checkpoint predicts nine quantiles for five future centered EMA levels.
We retain all outputs in an audited inference archive. Direct quantile crossings,
if present, are counted; the .5 output is used unchanged for the point path.
The null is filter catch-up for smoothed levels and zero actual change. Ridge_10
uses the established two-EMA-difference features and up to 252 mature examples,
fitted separately for each track. Foundation models use one EMA-level channel and
larger external training corpora: this is a comparison of full procedures rather
than an isolated architecture experiment. No failures are silently replaced.
'''),code('''
arrays,inference=foundation_quantiles(ROOT,signals,catalogue,protocol,recompute=RECOMPUTE)
ledger,fits,contexts,ridge_timing=replay(ROOT,signals,catalogue,protocol,arrays)
scores,macro=score_points(ledger)
for filename,table in [('historical_forecasts',ledger),('ridge_fit_audit',fits),('context_audit',contexts),
    ('point_scores',scores),('macro_scores',macro),('ridge_timing',ridge_timing)]:table.to_csv(OUT/f'{filename}.csv',index=False)
assert (fits.max_training_label_pos<=fits.origin_pos).all()
assert (contexts.context_end_pos==contexts.origin_pos).all()
assert ledger[ledger.role=='evaluation'].groupby(['dataset_id','track','model','horizon']).size().eq(40).all()
assert ledger[ledger.role=='latest'].actual.isna().all()
assert not ledger.duplicated(['dataset_id','origin_pos','track','model','horizon']).any()
print(len(ledger),'forecast rows;',len(fits),'mature Ridge fits;',len(contexts),'audited model contexts')
print('Native quantile crossings:',contexts.quantile_crossings.sum())
display(macro[(macro.horizon==5)&(macro.group=='stocks')].round(4))
'''),md('''
### 3. Compare the two tracks on the identical cohort
Skill = 1 − model loss / matched null loss, calculated within each dataset first,
then averaged with equal dataset weights. Positive skill is better. We never pool
log-price, temperature and CO2 errors. MAE uses median-like point losses; MSE is also
reported, but native q50 forecasts are not MSE-optimal means. Scores below concern
the fixed final forty origins per dataset and cannot be compared directly with
TS09/TS10 summaries that use other dates. Climate results remain a separate group.
'''),code('''
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for row,group in enumerate(['stocks','nonfinancial']):
    for col,track in enumerate(['smoothed','returns']):
        ax=axes[row,col]
        for model in order[1:]:
            f=macro[(macro.group==group)&(macro.track==track)&(macro.model==model)].sort_values('horizon')
            ax.plot(f.horizon,100*f.MAE_skill,'o-',label=labels[model],color=colors[model])
        ax.axhline(0,color='.4',ls='--');ax.set(title=f'{group}: {track}',ylabel='Equal-dataset MAE skill (%)',xlabel='Horizon (sessions / months)')
axes[0,0].legend(fontsize=8,ncol=2);fig.suptitle('Matched final 40 origins: positive skill beats the filter-aware null')
save_figure(fig,OUT,'01_macro_skill')
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,track in zip(axes,['smoothed','returns']):
    p=scores[(scores.group=='stocks')&(scores.track==track)&(scores.horizon==5)].pivot(index='dataset_id',columns='model',values='MAE_skill').reindex(columns=order[1:])
    cap=max(.05,float(np.abs(p.to_numpy()).max()))
    im=ax.imshow(100*p,cmap='RdBu',vmin=-100*cap,vmax=100*cap,aspect='auto')
    for i in range(len(p)):
        for j in range(len(p.columns)):ax.text(j,i,f'{100*p.iloc[i,j]:.0f}',ha='center',va='center',fontsize=9,color='black' if abs(p.iloc[i,j])<.6*cap else 'white')
    ax.set(yticks=range(len(p)),yticklabels=p.index,xticks=range(len(p.columns)),xticklabels=[labels[m] for m in p.columns],title=f'h=5 {track}: MAE skill (%)')
    ax.tick_params(axis='x',rotation=40);fig.colorbar(im,ax=ax,label='MAE skill (%)')
save_figure(fig,OUT,'02_stock_heterogeneity')
'''),md('''
### 4. Attach earlier-error intervals to BOTH tracks
For every evaluated or latest forecast, use the last 32 complete residual vectors
whose origin+5 is no later than the forecast origin. Then add empirical residual
10/50/90 percentiles to the point forecast. The interval is nominally 80%, with
coverage, width, interval score and pinball losses exported in original units.
The residual-adjusted median differs from the unadjusted point score above.
Small samples, temporal dependence and model misspecification preclude a coverage
guarantee. Monthly calibration/evaluation targets overlap heavily. Coverage alone
does not reward sharpness; read it alongside width and interval score.
'''),code('''
intervals,calibration,interval_scores=residual_intervals(ledger,protocol['residual_window'])
for filename,table in [('residual_intervals',intervals),('calibration_audit',calibration),('interval_scores',interval_scores)]:table.to_csv(OUT/f'{filename}.csv',index=False)
assert (calibration.last_calibration_label<=calibration.origin_pos).all()
assert calibration.calibration_n.eq(32).all()
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    p=interval_scores[(interval_scores.group=='stocks')&(interval_scores.track==track)&(interval_scores.horizon==5)].groupby('model').coverage80.mean().reindex(order)
    ax.bar(range(len(p)),100*p,color=[colors[m] for m in p.index]);ax.axhline(80,color='black',ls='--',label='Nominal 80%')
    ax.set(xticks=range(len(p)),xticklabels=[labels[m] for m in p.index],ylim=(0,100),ylabel='Mean stock coverage (%)',title=f'h=5 {track}: last-32-error intervals')
    ax.tick_params(axis='x',rotation=40)
    for j,v in enumerate(p):ax.text(j,100*v+1,f'{100*v:.1f}',ha='center',fontsize=9)
axes[0].legend();save_figure(fig,OUT,'03_residual_coverage')
display(interval_scores[(interval_scores.dataset_id=='SPY')&(interval_scores.horizon==5)].round(5))
'''),md('''
### 5. Read the paired uncertainty before ranking methods
At primary h=5, normalized MAE difference = (model MAE − null MAE)/null MAE.
Negative values favor the model. Resample contiguous blocks of 1, 4 and 8 origins,
using synchronized blocks across all six stocks, with equal asset weights and fixed
per-asset baseline denominators. Two thousand draws give exploratory 95% ranges.
No multiple-comparison correction or asset-population inference is performed. The
climate panel aligns relative positions, not shared calendar dates. Do not promote
the best result from this already examined panel into a validated trading model.
'''),code('''
ranges=paired_ranges(ledger,seed=protocol['seed']);ranges.to_csv(OUT/'paired_block_ranges.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,4.4))
for ax,track in zip(axes,['smoothed','returns']):
    p=ranges[(ranges.group=='stocks')&(ranges.track==track)&(ranges.block_origins==4)].set_index('model').reindex(order[1:])
    for j,(model,r) in enumerate(p.iterrows()):
        ax.plot([100*r.lower95,100*r.upper95],[j,j],color=colors[model],lw=3)
        ax.plot(100*r.normalized_MAE_difference,j,'o',color=colors[model])
    ax.axvline(0,color='black',ls='--');ax.set(yticks=range(len(p)),yticklabels=[labels[m] for m in p.index],xlabel='Normalized MAE difference (%)',title=f'h=5 stocks {track}: block 4 origins')
save_figure(fig,OUT,'04_paired_uncertainty')
display(ranges[(ranges.group=='stocks')&(ranges.block_origins==4)].round(4))
'''),md(r'''
### 6. A random-walk control distinguishes catch-up from prediction
Generate 128 independent Gaussian random-walk histories after checkpoint retrieval,
sigma .01, with no data-dependent model selection. Given each history, future raw
increments have conditional mean zero. For any fixed point forecast $m_h$, exact
return conditional MSE is $h\sigma^2+m_h^2$. The EMA null is its conditional mean;
EMA future variance is $\sigma^2\sum_{j=1}^h(1-.8^j)^2$.
The chart averages these **analytic conditional risks** over generated histories,
not noisy realized outcomes. A ratio above one is excess risk versus the known
conditional mean. This does not require the foundation forecast to fit zero exactly.
'''),code('''
control=martingale_control(arrays,protocol['seed']);control.to_csv(OUT/'martingale_control.csv',index=False)
assert control[['smoothed_conditional_MSE_ratio','return_conditional_MSE_ratio']].ge(1-1e-12).all().all()
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric in zip(axes,['smoothed_conditional_MSE_ratio','return_conditional_MSE_ratio']):
    for model,f in control.groupby('model'):ax.plot(f.horizon,f[metric],'o-',label=model.replace('chronos-bolt-',''))
    ax.axhline(1,color='black',ls='--',label='Known null');ax.set(title=metric.replace('_',' '),xlabel='Future step',ylabel='Exact conditional MSE / null MSE')
axes[0].legend();save_figure(fig,OUT,'05_martingale_control')
display(control[control.horizon==5].round(4))
'''),md('''
### 7. Compare inference cost with context and checkpoint size
The saved fresh-inference measurements use local CPU float32, one torch thread,
batch 32 and five forecast steps. Both tracks share one foundation inference, so
latency is amortized per origin for both. The Ridge measure includes two track
fits with mature training examples. Foundation inference excludes pretraining,
installation, downloads, model load, decoding/scoring and interval calibration;
load time and weight size are exported separately. This is one local run, with
different batching, not a hardware-general speed benchmark or peak-memory claim.
'''),code('''
compute=pd.DataFrame(inference['timings']);compute.to_csv(OUT/'foundation_compute.csv',index=False)
compute_summary=compute.groupby('model').agg(parameters=('parameters','first'),weights_bytes=('weights_bytes','first'),
    mean_ms=('milliseconds_per_origin_both_tracks','mean'),load_seconds=('load_seconds','first')).reindex(order[2:])
compute_summary.to_csv(OUT/'compute_summary.csv');display(compute_summary.round(3))
fig,ax=plt.subplots(figsize=(10,4))
for j,(model,r) in enumerate(compute_summary.iterrows()):ax.bar(j,r.mean_ms,color=colors[model])
ax.bar(4,ridge_timing.milliseconds_per_origin_both_tracks.mean(),color=colors['Ridge_10'])
ax.set(xticks=range(5),xticklabels=[labels[m] for m in compute_summary.index]+['Ridge fits'],
    ylabel='Milliseconds per origin, both tracks',title='Recorded CPU cost: amortized foundation inference versus two Ridge fits')
save_figure(fig,OUT,'06_compute_cost')
print('Recorded optional-inference environment:',inference['versions'])
'''),md('''
### 8. Audit future access and keep the latest outcomes unscored
Rebuild a stock signal through an earlier origin. Its context must exactly match
the full-history replay context. Perturb later data and require the same context.
Every Ridge training label and interval residual must have matured before use.
Latest rows deliberately have no known actual or performance score. They are a
reproducibility example from the frozen snapshot, not current trading advice.
'''),code('''
signal=signals['SPY'];t=int(ledger[(ledger.dataset_id=='SPY')&(ledger.role=='evaluation')].origin_pos.min())
full=SupervisedData(signal);prefix=SupervisedData(signal.iloc[:t+1])
changed=signal.copy();changed.iloc[t+1:]+=3;perturbed=SupervisedData(changed)
for length in protocol['contexts']:
    np.testing.assert_array_equal(centered_context(full.smoothed,t,length),centered_context(prefix.smoothed,t,length))
    np.testing.assert_array_equal(centered_context(full.smoothed,t,length),centered_context(perturbed.smoothed,t,length))
latest=ledger[ledger.role=='latest'].copy();latest.to_csv(OUT/'latest_unscored_forecasts.csv',index=False)
checks={'prefix_context_exact':True,'future_perturbation_context_exact':True,
    'all_ridge_labels_mature':bool((fits.max_training_label_pos<=fits.origin_pos).all()),
    'all_calibration_labels_mature':bool((calibration.last_calibration_label<=calibration.origin_pos).all()),
    'latest_all_unscored':bool(latest.actual.isna().all()),'all_evaluation_cells_n40':True,
    'no_model_selection':True,'pretraining_independence_established':False}
(OUT/'causality_checks.json').write_text(json.dumps(checks,indent=2));display(pd.Series(checks))
'''),md('''
## Checks and interpretation
Read equal-dataset MAE/MSE skills, asset heterogeneity, paired ranges, interval
coverage/width/scores and computational costs together. A larger checkpoint or
longer context is not automatically an improvement. EMA inversion can magnify
errors in the return track even when smoothed trajectories look convincing.
Chronological input integrity cannot establish independence from pretraining.

## Program completion
TS00–TS11 now cover the original twelve-notebook roadmap. The two foundation
checkpoints are real pretrained demonstrations, not randomly initialized stand-ins.
Optional larger or different families, fine-tuning and covariate models remain
future extensions; they are not prerequisites for completing the planned laboratory.
No historical result here establishes prospective stock-return skill. A future
deployment still requires a frozen rule, archived pre-outcome forecasts, costs,
uncertainty and the TS10 timing/eligibility checks.

### Sources
- [Chronos-Bolt tiny model card](https://huggingface.co/amazon/chronos-bolt-tiny)
- [Chronos-Bolt mini model card](https://huggingface.co/amazon/chronos-bolt-mini)
- [Official implementation and API](https://github.com/amazon-science/chronos-forecasting)
- [Chronos paper](https://arxiv.org/abs/2403.07815)
The exact model-card bytes, configuration, weights, commit revisions and license
are bundled under `inputs/foundation_models`. Vendor performance claims are not
used as evidence of performance on this project's panel.
'''),code(r'''
primary=macro[(macro.group=='stocks')&(macro.horizon==5)&(macro.model.isin(['chronos-bolt-tiny_L256','chronos-bolt-mini_L256']))]
parts=[]
for track in ['smoothed','returns']:
    p=primary[primary.track==track].set_index('model')
    parts.append(f"{track}: Tiny 256 **{100*p.loc['chronos-bolt-tiny_L256','MAE_skill']:.2f}%**, Mini 256 **{100*p.loc['chronos-bolt-mini_L256','MAE_skill']:.2f}%** MAE skill versus the matched null")
summary='## Observed results\n\n'+'. '.join(parts)+'. Each result averages six stocks, with 40 matched scored origins per stock at h=5. Positive skill is better. Both contexts, both target tracks and both climate datasets are retained in the detailed results. Pretraining overlap is unknown; these historical scores do not establish prospective trading skill.'
(OUT/'observed_summary.md').write_text(summary);display(Markdown(summary))
manifest=write_manifest(OUT,protocol,{'catalogue':'inputs/cross_dataset/catalogue.json',
    'checkpoint_manifest':'inputs/foundation_models/checkpoint_manifest.json'},tables=[p.name for p in sorted(OUT.glob('*.csv'))],seed=protocol['seed'])
manifest.update(scope='TS11_pinned_foundation_demonstration',claims='Historical fixed-protocol comparison; unknown pretraining overlap; no trading-skill promotion',
    inference_manifest_sha256=digest(OUT/'inference_manifest.json'),inference_cache_sha256=digest(OUT/'foundation_quantiles.npz'),
    protocol_sha256=digest(ROOT/'TS11_PROTOCOL.json'))
(OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
print('Program complete: 12 executed notebooks TS00–TS11; latest forecasts remain unscored.')
''')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'},'milestone':8})
nbf.validate(nb);path=ROOT/'notebooks'/f'{name}.ipynb';nbf.write(nb,path);print(path)
