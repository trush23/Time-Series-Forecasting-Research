"""Frozen representative-model benchmarks with dataset and calendar isolation.

The signals may be log adjusted prices or physical levels. 'returns' is the legacy
track identifier for actual future signal CHANGE; physical units stay explicit.
"""
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from .data import digest,validate_series
from .classical import ModelSpec
from .prequential import OriginRunner,choose_models
from .learned_evaluation import LearnedRunner
from .tabular import TabularSpec
from .neural import NeuralSpec
from .supervised import FeatureSpec
from .evaluation import scorecard,paired_block_difference


@dataclass(frozen=True)
class CrossSpec:
    outer_starts: tuple = (800,1000,1200)
    outer_step: int = 10
    inner_count: int = 6
    inner_step: int = 20
    max_samples: int = 252
    seeds: tuple = (1731,1732,1733)
    min_success: float = .8
    seasonal_period: int = 0
    neural_epochs: int = 30

    def __post_init__(self):
        if not self.outer_starts or tuple(sorted(set(self.outer_starts)))!=tuple(self.outer_starts):
            raise ValueError('Need increasing unique outer starts')
        if any(isinstance(x,bool) or not isinstance(x,int) or x<1 for x in [*self.outer_starts,self.outer_step,self.inner_count,self.inner_step,self.max_samples,self.neural_epochs]):
            raise ValueError('Positive integer schedules and budgets required')
        if self.max_samples<40 or self.seasonal_period not in (0,12) or not 0<=self.min_success<=1 or not self.seeds or len(set(self.seeds))!=len(self.seeds):
            raise ValueError('Invalid training, seasonal or selection settings')


def load_cross_inputs(folder):
    folder=Path(folder);catalogue=json.loads((folder/'catalogue.json').read_text());signals={}
    if len({m['dataset_id'] for m in catalogue})!=len(catalogue):raise ValueError('Duplicated dataset identity')
    stock_index=None
    for listed in catalogue:
        name=listed['dataset_id'];meta=json.loads((folder/f'{name}.metadata.json').read_text())
        if listed!=meta or digest(folder/f'{name}.csv')!=meta['sha256']:raise ValueError('Dataset provenance mismatch: '+name)
        f=pd.read_csv(folder/f'{name}.csv',parse_dates=['Date']).set_index('Date')
        signal=validate_series(f.value.astype(float))
        if len(signal)!=meta['rows'] or str(signal.index[0].date())!=meta['first'] or str(signal.index[-1].date())!=meta['last']:
            raise ValueError('Dataset shape/calendar mismatch')
        if meta['group']=='stocks':
            if stock_index is not None and not signal.index.equals(stock_index):raise ValueError('Stock panel calendars differ; no implicit intersection/fill')
            stock_index=signal.index
        signals[name]=signal
    return signals,catalogue


def seasonal_changes(channels,period=12,alpha=.2):
    """Repeat observed fast-EMA differences; invert EMA dynamics for signal changes.

    All future differences are generated, never teacher forced. No raw predictors.
    This is seasonal naive on EMA CHANGES, not a seasonal naive raw-level forecast.
    """
    v=np.asarray(channels,dtype=float)
    if v.ndim!=2 or v.shape[1]!=2 or len(v)<period or not np.isfinite(v[-period:]).all():
        raise ValueError('Need finite seasonal smoothed histories')
    predicted=v[-period:,0][:5]
    raw=(predicted-(1-alpha)*np.r_[v[-1,0],predicted[:-1]])/alpha
    return np.cumsum(predicted),np.cumsum(raw)


class CrossRunner:
    def __init__(self,signal,spec=CrossSpec(),checkpoint_dir=None):
        self.signal,self.spec=validate_series(signal),spec
        self.classical=OriginRunner(signal,[ModelSpec('AR5_changes','ar_diff',lags=5)],train_window=spec.max_samples)
        self.tabular=LearnedRunner(signal,[TabularSpec('Ridge_10','ridge',10.),TabularSpec('HistBoost','boost')],max_samples=spec.max_samples)
        self.neural=LearnedRunner(signal,[NeuralSpec('GRU','gru',max_epochs=spec.neural_epochs),
            NeuralSpec('NBEATS_demo','nbeats_demo',max_epochs=spec.neural_epochs)],mode='neural',
            max_samples=spec.max_samples,seeds=spec.seeds,checkpoint_dir=checkpoint_dir)
        self.names=['AR5_changes','Ridge_10','HistBoost','GRU','NBEATS_demo']
        if spec.seasonal_period:self.names.append('Seasonal_EMA_changes')

    def predictions(self,t):
        tab=self.tabular.predictions(t)
        classical=self.classical.predictions(t)
        neural=self.neural.predictions(t)
        parts=[tab,classical[classical.model=='AR5_changes'],neural[neural.model.isin(['GRU','NBEATS_demo'])]]
        if self.spec.seasonal_period:
            ds,raw=seasonal_changes(self.tabular.data.channels[:t+1],self.spec.seasonal_period)
            seasonal=tab[tab.model.isin(['filter_nochange','zero'])].copy()
            seasonal['prediction']=[self.tabular.data.smoothed.iloc[t]+ds[h-1] if track=='smoothed' else raw[h-1]
                                    for track,h in zip(seasonal.track,seasonal.horizon)]
            seasonal['model']='Seasonal_EMA_changes';seasonal['chosen_model']='Seasonal_EMA_changes';seasonal['fit_status']='ok'
            parts.append(seasonal)
        result=pd.concat(parts,ignore_index=True)
        assert not result.duplicated(['track','model','horizon']).any()
        return result

    def labels(self,frame):return self.tabular.attach_labels(frame)


def nested_cross(signal,spec=CrossSpec(),checkpoint_dir=None,progress=False):
    if spec.outer_starts[-1]>=len(signal)-5:raise ValueError('No complete final evaluation block')
    first=spec.outer_starts[0]-6-(spec.inner_count-1)*spec.inner_step
    if first<205:raise ValueError('Insufficient history for mature inner examples')
    runner=CrossRunner(signal,spec,checkpoint_dir)
    out,inner_frames,selections,tuning=[],[],[],[]
    for block,start in enumerate(spec.outer_starts):
        inner=[start-6-k*spec.inner_step for k in reversed(range(spec.inner_count))]
        records=runner.labels(pd.concat([runner.predictions(t) for t in inner],ignore_index=True))
        cutoff=signal.index[start-1]
        assert (records.available_at<=cutoff).all()
        choices,table=choose_models(records,runner.names,spec.min_success)
        table['block']=block;tuning.append(table);records['block']=block;inner_frames.append(records)
        stop=spec.outer_starts[block+1] if block+1<len(spec.outer_starts) else len(signal)-5
        origins=list(range(start,stop,spec.outer_step))
        if block==len(spec.outer_starts)-1:origins.append(len(signal)-1)
        for track,name in choices.items():selections.append({'block':block,'track':track,'chosen_model':name,
            'selection_cutoff':cutoff,'outer_first_origin':signal.index[start],'inner_last_label':signal.index[inner[-1]+5],
            'objective':'equal-horizon relative MAE','minimum_success_fraction':spec.min_success})
        for t in origins:
            frame=runner.predictions(t)
            selected=pd.concat([frame[(frame.track==track)&(frame.model==model)] for track,model in choices.items()]).copy()
            selected['chosen_model']=selected.model;selected['model']='selected_pipeline'
            frame=pd.concat([frame,selected],ignore_index=True);frame['block']=block;frame['selection_cutoff']=cutoff
            out.append(runner.labels(frame))
        if progress:print(f'Block {block+1}/3: {choices}',flush=True)
    result={'forecasts':pd.concat(out,ignore_index=True),'inner_forecasts':pd.concat(inner_frames,ignore_index=True),
        'selections':pd.DataFrame(selections),'tuning':pd.concat(tuning,ignore_index=True),
        'classical_audit':pd.DataFrame(runner.classical.audit),'tabular_audit':pd.DataFrame(runner.tabular.audit),
        'neural_audit':pd.DataFrame(runner.neural.audit),'learning_curves':pd.DataFrame(runner.neural.traces),
        'seed_forecasts':pd.DataFrame(runner.neural.members)}
    result['scores']=scorecard(result['forecasts'])
    interval=[]
    for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
        for length in [1,2,4]:interval.append({'track':track,**paired_block_difference(result['forecasts'],track,'selected_pipeline',baseline,5,length,2000)})
    result['paired_uncertainty']=pd.DataFrame(interval)
    slices=[]
    for block,frame in result['forecasts'].groupby('block'):
        card=scorecard(frame);card['block']=block;slices.append(card)
    result['block_scores']=pd.concat(slices,ignore_index=True)
    return result


def macro_scores(scores,catalogue):
    """Equal-dataset average of within-dataset skills; never pool physical-unit errors."""
    f=scores.merge(pd.DataFrame(catalogue)[['dataset_id','group']],on='dataset_id',validate='many_to_one')
    return f.groupby(['group','track','horizon','model']).agg(datasets=('dataset_id','nunique'),
        mean_MAE_skill=('MAE_skill','mean'),mean_MSE_skill=('MSE_skill','mean'),
        median_MSE_skill=('MSE_skill','median'),min_MSE_skill=('MSE_skill','min'),max_MSE_skill=('MSE_skill','max'),
        positive_datasets=('MSE_skill',lambda x:int((x>0).sum()))).reset_index()


def synchronized_panel_intervals(records,stock_ids,draws=2000,seed=1731):
    """Paired time-block resampling synchronized across the correlated stock panel.

    Equal asset weights; each absolute-loss difference divided by that asset's
    full-sample baseline MAE. Denominators are held fixed in this conditional audit.
    No IID asset bootstrap, population-of-assets inference or multiplicity correction.
    """
    rows=[];rng=np.random.default_rng(seed)
    for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
        arrays=[];reference=None
        for identifier in stock_ids:
            f=records[(records.dataset_id==identifier)&(records.track==track)&(records.horizon==5)&(records.status=='scored')]
            p=f.pivot(index='origin',columns='model',values='error').sort_index()
            if reference is not None and not p.index.equals(reference):raise ValueError('Panel origin mismatch')
            reference=p.index;den=p[baseline].abs().mean()
            if not den>0:raise ValueError('Degenerate panel normalization')
            arrays.append(((p.selected_pipeline.abs()-p[baseline].abs())/den).to_numpy())
        delta=np.mean(arrays,axis=0)
        for length in [1,2,4]:
            if length>len(delta):raise ValueError('Insufficient panel origins')
            starts=rng.integers(0,len(delta)-length+1,size=(draws,int(np.ceil(len(delta)/length))))
            idx=(starts[...,None]+np.arange(length)).reshape(draws,-1)[:,:len(delta)]
            means=delta[idx].mean(axis=1)
            rows.append({'track':track,'assets':len(stock_ids),'n_origins':len(delta),'block_origins':length,
                'draws':draws,'mean_normalized_MAE_difference':float(delta.mean()),
                'lower95':float(np.quantile(means,.025)),'upper95':float(np.quantile(means,.975))})
    return pd.DataFrame(rows)
