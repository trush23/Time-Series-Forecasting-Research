"""Exact NB9 import/replay and sample-wise next-close target adapters."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import digest


def delayed_samples(native_samples,max_steps=5):
    """Map joint t-origin cumulative/level draws to changes from t+1.

    Difference EACH draw before quantiling. Marginal quantiles cannot be subtracted.
    Five native steps support four holding intervals after next-close entry.
    """
    a=np.asarray(native_samples,dtype=float)
    if a.ndim!=2 or a.shape[1]!=max_steps or max_steps!=5 or not np.isfinite(a).all():
        raise ValueError('Need finite joint draws with five native horizons')
    return a[:,1:]-a[:,:1]


def summarize_sample(values,weights=None,actual=None):
    a=np.asarray(values,dtype=float)
    w=np.ones(len(a))/len(a) if weights is None else np.asarray(weights,dtype=float)
    if a.ndim!=1 or len(a)==0 or w.shape!=a.shape or not np.isfinite(a).all() or not np.isfinite(w).all() or np.any(w<0) or not w.sum()>0:
        raise ValueError('Need finite samples and nonnegative weights')
    order=np.argsort(a);a,w=a[order],w[order];w=w/w.sum();cdf=np.cumsum(w)
    q=a[np.minimum(np.searchsorted(cdf,[.1,.5,.9]),len(a)-1)]
    result={'mean':float(w@a),'q10':q[0],'median':q[1],'q90':q[2],'p_up':float(w[a>0].sum())}
    if actual is not None:
        pair=np.sum(w*(a*(np.cumsum(w)-w)-(np.cumsum(w*a)-w*a)))
        result.update(CRPS=float(w@np.abs(a-actual)-pair),squared_error=float((result['mean']-actual)**2),
            absolute_error=float(abs(result['mean']-actual)),coverage80=float(q[0]<=actual<=q[2]),width80=float(q[2]-q[0]))
    return result


class NB9Reference:
    """Frozen legacy features/metric/neighbor count; no retrospective reselection."""
    def __init__(self,folder):
        self.folder=Path(folder)
        self.manifest=json.loads((self.folder/'reference_manifest.json').read_text())
        for name,meta in self.manifest['files'].items():
            if digest(self.folder/name)!=meta['sha256']:raise ValueError('Reference bytes changed: '+name)
        self.source=json.loads((self.folder/'nb3/run_manifest.json').read_text())
        self.config=json.loads((self.folder/'nb9/forecast_configuration.json').read_text())
        self.market=pd.read_csv(self.folder/'nb3/market_ohlcv_snapshot.csv',index_col=0,parse_dates=True,float_precision='round_trip')
        states=pd.read_csv(self.folder/'nb3/interpretable_states.csv',index_col=0,parse_dates=True,float_precision='round_trip').drop(columns='segment')
        sc=self.source['scalers']
        physical=states.mul(pd.Series(sc['interpretable_scale'])).add(pd.Series(sc['interpretable_center']))
        self.index=states.index;self.positions=self.market.index.get_indexer(self.index)
        if np.any(self.positions<0) or not np.all(np.diff(self.positions)==1):raise ValueError('NB9 state/market alignment mismatch')
        self.q=(physical[self.config['features']].to_numpy()-np.array(self.config['center']))/np.array(self.config['scale'])
        self.x=np.log(self.market.Close)
        self.targets=np.column_stack([(self.x.shift(-(h+1))-self.x.shift(-1)).iloc[self.positions].to_numpy() for h in range(1,6)])
        self.forecasts=pd.read_csv(self.folder/'nb9/forecasts.csv',parse_dates=['date','label_end'],float_precision='round_trip')
        self.calibration_end=pd.Timestamp(json.loads((self.folder/'nb9/run_manifest.json').read_text())['calibration_end'])
        self.neighbor_cache={}

    def neighbors(self,t):
        if t in self.neighbor_cache:return self.neighbor_cache[t]
        p=self.positions;cfg=self.config
        ids=np.arange(t);ids=ids[(p[ids]+6<=p[t])&(p[ids]<p[t]-cfg['theiler'])]
        dist=np.linalg.norm(self.q[ids]-self.q[t],axis=1)/np.sqrt(self.q.shape[1]);chosen=[];ds=[]
        for j in np.argsort(dist):
            if not chosen or np.min(abs(p[ids[j]]-p[np.array(chosen)]))>=cfg['neighbor_spacing']:
                chosen.append(int(ids[j]));ds.append(float(dist[j]))
                if len(chosen)==cfg['k']:break
        if len(chosen)!=cfg['k']:raise ValueError('Insufficient eligible separated legacy neighbors')
        chosen=np.asarray(chosen);d=np.asarray(ds)
        w=np.exp(-cfg['weight_theta']*d/max(d.mean(),1e-12));w/=w.sum()
        recent=ids[-250:]
        result={'analog':(chosen,w),'recent':(recent,np.ones(len(recent))/len(recent)),
                'historical':(ids,np.ones(len(ids))/len(ids))}
        supported=bool(d[-1]<=cfg['support_radius'] and 1/(w@w)>=cfg['minimum_neff'])
        self.neighbor_cache[t]=(result,supported,d[-1]);return self.neighbor_cache[t]

    def replay(self,smoothed=None):
        rows=[];audit=[]
        for date in sorted(self.forecasts.date.unique()):
            t=int(self.index.get_indexer([date])[0]);p=int(self.positions[t])
            groups,supported,radius=self.neighbors(t)
            for model,(ids,w) in groups.items():
                audit.append({'date':date,'model':model,'origin_pos':p,'latest_label_pos':int(np.max(self.positions[ids]+6)),
                    'latest_feature_pos':int(np.max(self.positions[ids])),'neighbor_count':len(ids),'supported':supported,'radius':radius})
                for h in range(1,6):
                    rows.append({'date':date,'label_end':self.market.index[p+1+h],'track':'returns','horizon':h,'model':model,
                        'origin_pos':p,'actual':self.targets[t,h-1],'supported':supported,'forecast_steps':h+1,
                        **summarize_sample(self.targets[ids,h-1],w,self.targets[t,h-1])})
                    if smoothed is not None and h<=4 and model in ['analog','recent']:
                        values=smoothed.to_numpy()[self.positions[ids]+1+h]-smoothed.to_numpy()[self.positions[ids]+1]
                        actual=smoothed.iloc[p+1+h]-smoothed.iloc[p+1]
                        rows.append({'date':date,'label_end':self.market.index[p+1+h],'track':'smoothed','horizon':h,'model':model,
                            'origin_pos':p,'actual':actual,'supported':supported,'forecast_steps':h+1,**summarize_sample(values,w,actual)})
        return pd.DataFrame(rows),pd.DataFrame(audit)
