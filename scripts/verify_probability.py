"""Independent audit of the saved TS08 ledger, probability bounds and score tables."""
import json,math
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import ndtr
from tslab.supervised import SupervisedData,FeatureSpec


def verify_probability(root,folder,market,manifest):
    cfg=manifest['config'];settings=cfg['probability'];window=settings['calibration_window']
    ledger=pd.read_csv(folder/'issued_forecast_ledger.csv',parse_dates=['origin','target_end','available_at'])
    audit=pd.read_csv(folder/'calibration_audit.csv')
    fits=pd.read_csv(folder/'fit_audit.csv')
    bounds=pd.read_csv(folder/'intervals.csv',parse_dates=['origin'])
    quantiles=pd.read_csv(folder/'quantiles.csv',parse_dates=['origin'])
    data=SupervisedData(market,FeatureSpec(**cfg['features']))
    assert (fits.max_training_label<=fits.origin_pos).all()
    assert (audit.calibration_last_label_pos<=audit.origin_pos).all()
    assert audit.calibration_n.eq(window).all()
    assert ledger.groupby(['origin_pos','track']).model.nunique().eq(2).all()
    assert len(audit)==bounds[['origin_pos','track','model']].drop_duplicates().shape[0]
    for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
        sub=ledger[ledger.track==track]
        pos=sub.origin_pos.to_numpy();hs=sub.horizon.to_numpy();ends=pos+hs
        valid=ends<len(market)
        actual=(data.smoothed.to_numpy()[ends[valid]] if track=='smoothed'
            else market.to_numpy()[ends[valid]]-market.to_numpy()[pos[valid]])
        np.testing.assert_allclose(sub.actual.to_numpy()[valid],actual,atol=1e-12,rtol=1e-12)
        assert sub.actual.to_numpy()[~valid].size==0 or np.isnan(sub.actual.to_numpy()[~valid]).all()
        null=sub[sub.model==baseline]
        expected=data.smoothed.to_numpy()[null.origin_pos]+data.anchors[null.origin_pos,null.horizon-1] if track=='smoothed' else 0.
        np.testing.assert_allclose(null.prediction,expected,atol=1e-12)
    # Verify every final-fit scaler and label pool independently from eligible examples.
    for row in fits[fits.fit_status=='ok'].itertuples():
        X,y,idx=data.training(row.origin_pos,row.track,settings['max_samples'])
        assert row.max_training_label==idx[-1]+5 and row.n_train==len(X)
        np.testing.assert_allclose(json.loads(row.feature_mean),X.reshape(len(X),-1).mean(axis=0),atol=1e-14)
        np.testing.assert_allclose(json.loads(row.target_mean),y.mean(axis=0),atol=1e-14)
    # Reconstruct every query from stored historical predictions, not in-sample model fits.
    checked_distribution=[]
    for (track,model),part in ledger.groupby(['track','model']):
        error=part.pivot(index='origin_pos',columns='horizon',values='error').sort_index()
        scale=part.pivot(index='origin_pos',columns='horizon',values='scale').sort_index()
        point=part.pivot(index='origin_pos',columns='horizon',values='prediction').sort_index()
        a=audit[(audit.track==track)&(audit.model==model)]
        b=bounds[(bounds.track==track)&(bounds.model==model)]
        q=quantiles[(quantiles.track==track)&(quantiles.model==model)]
        # Check scales against an independent RMS calculation for all ledger origins.
        hs=np.arange(1,6);factors=np.sqrt(hs) if track=='returns' else np.sqrt(np.cumsum((1-(1-data.spec.fast_alpha)**hs)**2))
        for t in scale.index:
            v=data.channels[t-settings['scale_window']+1:t+1,0]
            expected=max(np.sqrt(np.mean(v*v)),1e-6)*factors
            np.testing.assert_allclose(scale.loc[t],expected,atol=1e-14)
        for row in a.itertuples():
            t=row.origin_pos
            e=-error.loc[error.index+5<=t].dropna().tail(window)
            assert len(e)==window
            assert e.index[0]==row.calibration_first_origin_pos and e.index[-1]==row.calibration_last_origin_pos
            s=scale.loc[e.index].to_numpy();now=scale.loc[t].to_numpy();p=point.loc[t].to_numpy();values=e.to_numpy()
            samples=p+values;mean=samples.mean(axis=0);sd=np.maximum(values.std(axis=0,ddof=1),1e-8)
            current=b[b.origin_pos==t]
            for (method,level),group in current.groupby(['method','level']):
                group=group.sort_values('horizon');tail=(1-level)/2;k=math.ceil((window+1)*level)
                if method=='gaussian':lo=mean+norm.ppf(tail)*sd;hi=mean+norm.ppf(1-tail)*sd
                elif method=='empirical':lo,hi=np.quantile(samples,[tail,1-tail],axis=0,method='inverted_cdf')
                else:
                    if method=='rank_absolute':radius=np.sort(abs(values),axis=0)[k-1]
                    elif method=='rank_scaled':radius=np.sort(abs(values)/s,axis=0)[k-1]*now
                    else:radius=np.sort(np.max(abs(values)/s,axis=1))[k-1]*now
                    lo=p-radius;hi=p+radius
                np.testing.assert_allclose(group.lower,lo,atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose(group.upper,hi,atol=1e-12,rtol=1e-10)
            for method in ['gaussian','empirical','rank_absolute','rank_scaled']:
                tables=current[current.method==method]
                levels=sorted(tables.level.unique())
                median=tables[tables.level==levels[0]].sort_values('horizon')['median'].to_numpy()
                actual=tables[tables.level==levels[0]].sort_values('horizon').actual.to_numpy()
                wis=.5*abs(actual-median)
                for level in levels:
                    rows=tables[tables.level==level].sort_values('horizon')
                    lo,hi=rows.lower.to_numpy(),rows.upper.to_numpy()
                    interval=hi-lo+2/(1-level)*(np.maximum(lo-actual,0)+np.maximum(actual-hi,0))
                    wis+=(1-level)/2*interval
                wis/=len(levels)+.5
                crps=np.full(5,np.nan)
                if method=='gaussian':
                    z=(actual-mean)/sd
                    crps=sd*(z*(2*ndtr(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))
                elif method=='empirical':
                    ordered=np.sort(samples,axis=0);n=len(ordered)
                    crps=np.mean(abs(ordered-actual),axis=0)-np.sum((2*np.arange(1,n+1)-n-1)[:,None]*ordered,axis=0)/n**2
                for h in range(1,6):checked_distribution.append({'origin':pd.Timestamp(market.index[t]),
                    'track':track,'model':model,'method':method,'horizon':h,'WIS':wis[h-1],'CRPS':crps[h-1]})
            origin=pd.Timestamp(market.index[t]);query_q=q[q.origin==origin]
            for (method,tau),group in query_q.groupby(['method','tau']):
                predicted=mean+norm.ppf(tau)*sd if method=='gaussian' else np.quantile(samples,tau,axis=0,method='inverted_cdf')
                np.testing.assert_allclose(group.sort_values('horizon')['quantile'],predicted,atol=1e-12)
    good=bounds[bounds.status=='scored'].copy()
    np.testing.assert_array_equal(good.covered,((good.lower<=good.actual)&(good.actual<=good.upper)).astype(float))
    np.testing.assert_allclose(bounds.width,bounds.upper-bounds.lower,atol=1e-12)
    marginal=good[good.method!='joint_max_scaled']
    score=marginal.width+2/(1-marginal.level)*(np.maximum(marginal.lower-marginal.actual,0)+np.maximum(marginal.actual-marginal.upper,0))
    np.testing.assert_allclose(marginal.interval_score,score,atol=1e-12)
    keys=['track','model','method','horizon','level']
    expected=good.groupby(keys).agg(n=('covered','size'),coverage=('covered','mean'),mean_width=('width','mean'),mean_interval_score=('interval_score','mean'))
    saved=pd.read_csv(folder/'interval_scorecard.csv').set_index(keys).sort_index()
    np.testing.assert_allclose(saved,expected.sort_index(),atol=1e-12,equal_nan=True)
    err=quantiles.actual-quantiles['quantile']
    np.testing.assert_allclose(quantiles.pinball_loss,np.maximum(quantiles.tau*err,(quantiles.tau-1)*err),atol=1e-12,equal_nan=True)
    expected=pd.DataFrame(checked_distribution)
    keys=['origin','track','model','method','horizon']
    saved=pd.read_csv(folder/'distribution_scores.csv',parse_dates=['origin'])
    np.testing.assert_allclose(saved.set_index(keys).sort_index(),expected.set_index(keys).sort_index(),atol=1e-12,equal_nan=True)
    groups=['track','model','method','horizon']
    expected_card=expected.groupby(groups).agg(n=('WIS','count'),WIS=('WIS','mean'),CRPS=('CRPS','mean'))
    saved_card=pd.read_csv(folder/'distribution_scorecard.csv').set_index(groups).sort_index()
    np.testing.assert_allclose(saved_card,expected_card.sort_index(),atol=1e-12,equal_nan=True)
    paths=good.groupby(['origin','track','model','method','level']).covered.min().reset_index()
    expected_path=paths.groupby(['track','model','method','level']).agg(n=('covered','size'),joint_coverage=('covered','mean'))
    saved_path=pd.read_csv(folder/'joint_coverage.csv').set_index(['track','model','method','level']).sort_index()
    np.testing.assert_allclose(saved_path,expected_path.sort_index(),atol=1e-12)
    # Latest bootstrap rows must use one intact native residual vector across horizons.
    native=pd.read_csv(folder/'latest_empirical_samples.csv').set_index(['track','model','sample_id','horizon']).prediction_sample
    draws=pd.read_csv(folder/'latest_bootstrap_draws.csv')
    assert draws.groupby(['track','draw_id']).native_sample_id.nunique().eq(1).all()
    expected=[native.loc[(r.track,r.model,r.native_sample_id,r.horizon)] for r in draws.itertuples()]
    np.testing.assert_allclose(draws['sample'],expected,atol=1e-12)
    print('TS08: ledger labels/scalers, mature calibration, every bound/quantile, scores and vector samples passed.')
