"""Prequential residual distributions and rolling interval demonstrations.

Rank correction is implemented exactly; exchangeability is NOT asserted for
refitted, overlapping time-series residuals. No general coverage guarantee is made.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from .supervised import FeatureSpec,SupervisedData
from .tabular import TabularSpec,fit_tabular
from .targets import make_targets
from .evaluation import scorecard

LEVELS=(.5,.8,.95)
QUANTILES=(.025,.1,.25,.5,.75,.9,.975)
METHODS=('gaussian','empirical','rank_absolute','rank_scaled','joint_max_scaled')


def pinball(actual,quantile,tau):
    if not 0<tau<1:raise ValueError('Quantile probability must be in (0,1)')
    e=np.asarray(actual)-np.asarray(quantile)
    return np.maximum(tau*e,(tau-1)*e)


def interval_score(actual,lower,upper,level):
    if not 0<level<1:raise ValueError('Coverage level must be in (0,1)')
    y,l,u=np.broadcast_arrays(actual,lower,upper)
    if np.any(l>u):raise ValueError('Reversed interval')
    alpha=1-level
    return u-l+2/alpha*np.maximum(l-y,0)+2/alpha*np.maximum(y-u,0)


def weighted_interval_score(actual,median,lower,upper,levels=LEVELS):
    levels=np.asarray(levels,dtype=float)
    if levels.ndim!=1 or len(levels)==0 or np.any((levels<=0)|(levels>=1)):
        raise ValueError('Invalid central interval levels')
    lower,upper=np.asarray(lower),np.asarray(upper)
    if lower.shape!=upper.shape or lower.shape[0]!=len(levels):raise ValueError('Interval shape mismatch')
    total=.5*np.abs(np.asarray(actual)-median)
    for j,level in enumerate(levels):total=total+(1-level)/2*interval_score(actual,lower[j],upper[j],level)
    return total/(len(levels)+.5)


def empirical_crps(samples,actual):
    """Exact CRPS of the finite equally weighted empirical CDF, not fair-ensemble CRPS."""
    x=np.sort(np.asarray(samples,dtype=float),axis=0)
    if len(x)==0 or not np.isfinite(x).all():raise ValueError('Need finite samples')
    n=len(x)
    weights=(2*np.arange(1,n+1)-n-1).reshape((n,)+(1,)*(x.ndim-1))
    return np.mean(np.abs(x-actual),axis=0)-np.sum(weights*x,axis=0)/n**2


def gaussian_crps(mean,sd,actual):
    mean,sd,actual=np.broadcast_arrays(mean,sd,actual)
    if np.any(sd<=0) or not np.isfinite(sd).all():raise ValueError('Positive finite standard deviation required')
    z=(actual-mean)/sd
    return sd*(z*(2*ndtr(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))


def rank_radius(scores,level):
    """ceil((n+1)*level)-th order statistic; +inf if finite calibration is too small."""
    a=np.asarray(scores,dtype=float)
    if a.ndim<1 or len(a)==0 or not np.isfinite(a).all() or np.any(a<0) or not 0<level<1:
        raise ValueError('Need finite nonnegative calibration scores and level in (0,1)')
    k=math.ceil((len(a)+1)*level)
    return np.full(a.shape[1:],np.inf) if k>len(a) else np.sort(a,axis=0)[k-1]


def distribution_intervals(errors,prediction,past_scales,current_scale,levels=LEVELS):
    """Pure function: callers must enforce maturity before supplying calibration errors."""
    levels=np.asarray(levels,dtype=float)
    if levels.ndim!=1 or len(levels)==0 or np.any(~np.isfinite(levels)) or np.any((levels<=0)|(levels>=1)) or np.any(np.diff(levels)<=0):
        raise ValueError('Coverage levels must be finite, increasing and in (0,1)')
    e,p,s,now=map(lambda a:np.asarray(a,dtype=float),(errors,prediction,past_scales,current_scale))
    if (e.ndim!=2 or e.shape[1]!=5 or len(e)<2 or p.shape!=(5,) or s.shape!=e.shape or now.shape!=(5,)
        or not all(np.isfinite(a).all() for a in [e,p,s,now]) or np.any(s<=0) or np.any(now<=0)):
        raise ValueError('Need aligned finite errors/scales and five-horizon query')
    mean=p+e.mean(axis=0);sd=np.maximum(e.std(axis=0,ddof=1),1e-8)
    samples=p+e
    result={}
    for method in METHODS:
        lo=[];hi=[]
        for level in levels:
            tail=(1-level)/2
            if method=='gaussian':l,u=mean+norm.ppf(tail)*sd,mean+norm.ppf(1-tail)*sd
            elif method=='empirical':l,u=np.quantile(samples,[tail,1-tail],axis=0,method='inverted_cdf')
            else:
                if method=='rank_absolute':r=rank_radius(np.abs(e),level)
                elif method=='rank_scaled':r=rank_radius(np.abs(e)/s,level)*now
                else:r=rank_radius(np.max(np.abs(e)/s,axis=1),level)*now
                l,u=p-r,p+r
            lo.append(l);hi.append(u)
        median=(mean if method=='gaussian' else np.quantile(samples,.5,axis=0,method='inverted_cdf') if method=='empirical' else p)
        result[method]={'lower':np.asarray(lo),'upper':np.asarray(hi),'median':median}
    return result,{'mean':mean,'sd':sd,'samples':samples}


@dataclass(frozen=True)
class ProbabilitySpec:
    forecast_start: int = 650
    evaluation_start: int = 1000
    evaluation_step: int = 5
    max_samples: int = 252
    calibration_window: int = 126
    scale_window: int = 60

    def __post_init__(self):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in self.__dict__.values()):
            raise ValueError('Budgets and positions must be positive integers')
        if self.max_samples<40 or self.calibration_window<40 or self.scale_window<10:
            raise ValueError('Insufficient training/calibration/scale budget')
        if self.evaluation_start<self.forecast_start+self.calibration_window+5:
            raise ValueError('Need matured calibration forecasts before evaluation')


def scale_proxy(data,t,track,window=60):
    """Positive smoothed-input RMS proxy; not a fitted conditional standard deviation."""
    v=data.channels[max(0,t-window+1):t+1,0]
    if len(v)!=window or not np.isfinite(v).all():raise ValueError('Need a mature scale window')
    rms=max(float(np.sqrt(np.mean(v*v))),1e-6)
    hs=np.arange(1,6)
    factors=np.sqrt(hs) if track=='returns' else np.sqrt(np.cumsum((1-(1-data.spec.fast_alpha)**hs)**2))
    return rms*factors


def point_ledger(log_price,spec=ProbabilitySpec(),feature_spec=FeatureSpec()):
    """Issue fixed Ridge_10 and null forecasts daily; labels attached afterward."""
    data=SupervisedData(log_price,feature_spec)
    rows,audits=[],[]
    model=TabularSpec('Ridge_10','ridge',10.)
    for t in range(spec.forecast_start,len(log_price)):
        for track in ['smoothed','returns']:
            X,y,origins=data.training(t,track,spec.max_samples)
            try:
                fit,detail=fit_tabular(X,y,model)
                residual=fit.predict(data.features[t]);status='ok';error=''
            except Exception as exc:
                residual=np.zeros(5);detail={};status='fallback';error=f'{type(exc).__name__}: {exc}'
            point=data.decode(residual,t,track)
            base=data.decode(np.zeros(5),t,track)
            scale=scale_proxy(data,t,track,spec.scale_window)
            baseline='filter_nochange' if track=='smoothed' else 'zero'
            audits.append({'origin_pos':t,'origin':log_price.index[t],'track':track,'model':model.name,
                'fit_status':status,'error':error,'max_training_label':int(origins[-1]+5),
                'first_training_origin':int(origins[0]),'last_training_origin':int(origins[-1]),**detail})
            for name,prediction,fit_status in [(model.name,point,status),(baseline,base,'baseline')]:
                for h in range(1,6):rows.append({'origin_pos':t,'origin':log_price.index[t],'track':track,
                    'horizon':h,'model':name,'prediction':prediction[h-1],'scale':scale[h-1],
                    'fit_status':fit_status,'training_cutoff':log_price.index[t],
                    'data_available_through':log_price.index[t]})
    ledger=pd.DataFrame(rows).merge(make_targets(log_price,data.smoothed).drop(columns='origin_pos'),
        on=['origin','track','horizon'],validate='many_to_one')
    ledger['filter_id']=f'ema_alpha_{feature_spec.fast_alpha}'
    ledger['error']=ledger.prediction-ledger.actual
    ledger['status']=np.where(ledger.actual.notna(),'scored','immature')
    ledger['units']=np.where(ledger.track=='returns','log_return','log_level')
    return ledger,pd.DataFrame(audits),data


def calibration_slice(part,t,window):
    """Return last window complete five-error vectors with outcome available by t.

    Conservative common maturity: use origins <= t-5 for EVERY horizon, including h=1.
    Future labels may exist in a retrospective ledger but are excluded explicitly.
    """
    past=part[part.origin_pos+5<=t]
    errors=past.pivot(index='origin_pos',columns='horizon',values='error').sort_index()
    errors=errors.reindex(columns=range(1,6)).dropna().tail(window)
    if len(errors)!=window:raise ValueError('Insufficient matured forecast residual vectors')
    scales=past.pivot(index='origin_pos',columns='horizon',values='scale').loc[errors.index,range(1,6)]
    # Point-ledger error is prediction-actual; distribution residual is actual-prediction.
    return -errors.to_numpy(),scales.to_numpy(),errors.index.to_numpy()


def probabilistic_replay(ledger,spec=ProbabilitySpec(),levels=LEVELS,quantiles=QUANTILES):
    """Build distributions before scoring outcomes; never tune a method on outer loss."""
    quantiles=np.asarray(quantiles,dtype=float)
    if quantiles.ndim!=1 or len(quantiles)==0 or not np.isfinite(quantiles).all() or np.any((quantiles<=0)|(quantiles>=1)) or np.any(np.diff(quantiles)<=0):
        raise ValueError('Quantile probabilities must be increasing and in (0,1)')
    positions=np.sort(ledger.origin_pos.unique());last=int(positions[-1])
    if spec.evaluation_start>last-5:raise ValueError('No fully mature evaluation origin')
    # Complete-horizon cohort + latest (wholly unscored) origin; partial tail omitted.
    schedule=list(range(spec.evaluation_start,last-4,spec.evaluation_step))
    if last not in schedule:schedule.append(last)
    rows,qs,dist_rows,audits,latest=[],[],[],[],[]
    for (track,model),part in ledger.groupby(['track','model']):
        queries=part.set_index(['origin_pos','horizon'])
        for t in schedule:
            query=queries.loc[t].sort_index()
            e,s,idx=calibration_slice(part,t,spec.calibration_window)
            prediction=query.prediction.to_numpy();now=query.scale.to_numpy()
            methods,dist=distribution_intervals(e,prediction,s,now,levels)
            origin=query.origin.iloc[0]
            audits.append({'origin':origin,'origin_pos':t,'track':track,'model':model,
                'calibration_n':len(e),'calibration_first_origin_pos':int(idx[0]),
                'calibration_last_origin_pos':int(idx[-1]),'calibration_last_label_pos':int(idx[-1]+5),
                'scale_available_pos':t})
            for method,values in methods.items():
                for j,level in enumerate(levels):
                    for h in range(1,6):
                        actual=float(query.actual.iloc[h-1]);l=values['lower'][j,h-1];u=values['upper'][j,h-1]
                        rows.append({'origin':origin,'origin_pos':t,'track':track,'model':model,'method':method,
                            'horizon':h,'level':level,'lower':l,'upper':u,'median':values['median'][h-1],
                            'actual':actual,'status':'scored' if np.isfinite(actual) else 'immature',
                            'covered':float(l<=actual<=u) if np.isfinite(actual) else np.nan,'width':u-l,
                            'interval_score':float(interval_score(actual,l,u,level)) if method!='joint_max_scaled' else np.nan,
                            'calibration_n':len(e),'calibration_last_label_pos':int(idx[-1]+5)})
                if method!='joint_max_scaled':
                    wis=weighted_interval_score(query.actual.to_numpy(),values['median'],values['lower'],values['upper'],levels)
                    for h in range(1,6):dist_rows.append({'origin':origin,'track':track,'model':model,'method':method,
                        'horizon':h,'WIS':wis[h-1],'CRPS':float(gaussian_crps(dist['mean'][h-1],dist['sd'][h-1],query.actual.iloc[h-1])) if method=='gaussian'
                            else float(empirical_crps(dist['samples'][:,h-1],query.actual.iloc[h-1])) if method=='empirical' else np.nan})
            for method in ['gaussian','empirical']:
                for tau in quantiles:
                    values=dist['mean']+norm.ppf(tau)*dist['sd'] if method=='gaussian' else np.quantile(dist['samples'],tau,axis=0,method='inverted_cdf')
                    for h in range(1,6):qs.append({'origin':origin,'track':track,'model':model,'method':method,'horizon':h,
                        'tau':tau,'quantile':values[h-1],'actual':query.actual.iloc[h-1],
                        'pinball_loss':float(pinball(query.actual.iloc[h-1],values[h-1],tau))})
            if t==last:
                for j,sample in enumerate(dist['samples']):
                    for h in range(1,6):latest.append({'origin':origin,'track':track,'model':model,'sample_id':j,
                        'calibration_origin_pos':int(idx[j]),'horizon':h,'prediction_sample':sample[h-1]})
    intervals=pd.DataFrame(rows);quantile_frame=pd.DataFrame(qs);distribution_scores=pd.DataFrame(dist_rows)
    scored=intervals[intervals.status=='scored']
    keys=['track','model','method','horizon','level']
    card=scored.groupby(keys).agg(n=('covered','size'),coverage=('covered','mean'),
        mean_width=('width','mean'),mean_interval_score=('interval_score','mean')).reset_index()
    path=scored.groupby(['origin','track','model','method','level']).covered.agg(['min','size']).reset_index()
    assert path['size'].eq(5).all()
    path_card=path.groupby(['track','model','method','level']).agg(n=('min','size'),joint_coverage=('min','mean')).reset_index()
    score_card=distribution_scores.groupby(['track','model','method','horizon']).agg(n=('WIS','count'),WIS=('WIS','mean'),CRPS=('CRPS','mean')).reset_index()
    quantile_card=quantile_frame.groupby(['track','model','method','horizon','tau']).agg(n=('pinball_loss','count'),pinball_loss=('pinball_loss','mean')).reset_index()
    forecast=ledger[ledger.origin_pos.isin(schedule)].copy()
    return {'historical_forecasts':forecast,'intervals':intervals,'interval_scorecard':card,'joint_coverage':path_card,
        'quantiles':quantile_frame,'quantile_scorecard':quantile_card,'distribution_scores':distribution_scores,
        'distribution_scorecard':score_card,'calibration_audit':pd.DataFrame(audits),
        'latest_empirical_samples':pd.DataFrame(latest),'point_scorecard':scorecard(forecast)}


def simulation_controls(seed=1731,repetitions=200,calibration_n=126,test_n=500):
    """Independent Gaussian innovation vectors; deliberate 3x scale-shift failure case.

    Oracle scaling is an explanatory control, not the stock-data scale estimator.
    Rows/vectors in this control are independent; stock replay origins are not.
    """
    if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in [repetitions,calibration_n,test_n]) or calibration_n<40:
        raise ValueError('Need positive integer simulation budgets and >=40 calibration vectors')
    rng=np.random.default_rng(seed);rows=[];alpha=.2
    for rep in range(repetitions):
        innovations=rng.normal(size=(calibration_n+test_n,5))
        actual_returns=innovations.cumsum(axis=1)
        smooth_errors=np.column_stack([innovations[:,:h]@(1-(1-alpha)**np.arange(h,0,-1)) for h in range(1,6)])
        for track,e in [('smoothed',smooth_errors),('returns',actual_returns)]:
            calibration=e[:calibration_n];test=e[calibration_n:]
            for case,factor in [('IID',1.),('Abrupt 3x scale shift',3.)]:
                actual=test*factor
                for level in LEVELS:
                    raw=rank_radius(np.abs(calibration),level)
                    scaled=raw*factor
                    maximum=rank_radius(np.max(np.abs(calibration)/np.std(calibration,axis=0,ddof=1),axis=1),level)
                    joint_radius=maximum*np.std(calibration,axis=0,ddof=1)*factor
                    for method,radius in [('rank_absolute',raw),('oracle_scaled_rank',scaled),('oracle_scaled_joint_max',joint_radius)]:
                        covered=np.abs(actual)<=radius
                        rows.append({'replicate':rep,'track':track,'case':case,'method':method,'level':level,
                            'marginal_coverage':covered.mean(),'joint_coverage':covered.all(axis=1).mean(),
                            'mean_width':float(2*np.mean(radius))})
    return pd.DataFrame(rows)
