"""Independent saved-artifact audit of TS10's clocks, distributions and cash ledger."""
import json
import numpy as np
import pandas as pd
from tslab.data import digest
from tslab.supervised import SupervisedData,FeatureSpec


def verify_trading_integration(root,folder,manifest):
    protocol=json.loads((root/'TS10_PROTOCOL.json').read_text());assert manifest['config']==protocol
    assert digest(root/'TS10_PROTOCOL.json')==manifest['source']['protocol_sha256']
    ref=root/'inputs/nb_integration';meta=json.loads((ref/'reference_manifest.json').read_text())
    assert digest(ref/'reference_manifest.json')==protocol['reference_manifest_sha256']
    for name,info in meta['files'].items():assert digest(ref/name)==info['sha256']
    market=pd.read_csv(ref/'nb3/market_ohlcv_snapshot.csv',index_col=0,parse_dates=True,float_precision='round_trip')
    price=market.Close;signal=np.log(price);data=SupervisedData(signal,FeatureSpec(**protocol['features']))
    native=pd.read_csv(folder/'native_issued_forecast_ledger.csv');forecasts=pd.read_csv(folder/'delayed_forecasts.csv',parse_dates=['date','label_end','target_start'])
    fits=pd.read_csv(folder/'native_fit_audit.csv');audit=pd.read_csv(folder/'delayed_calibration_audit.csv',parse_dates=['date'])
    assert (fits.max_training_label<=fits.origin_pos).all()
    assert (audit.calibration_last_label_pos<=audit.origin_pos).all() and audit.calibration_n.eq(126).all()
    cache={}
    for row in fits[fits.fit_status=='ok'].itertuples():
        key=(row.origin_pos,row.track)
        X,y,ids=data.training(row.origin_pos,row.track,252)
        assert row.n_train==len(X) and row.max_training_label==ids[-1]+5
        np.testing.assert_allclose(json.loads(row.feature_mean),X.reshape(len(X),-1).mean(axis=0),atol=1e-13)
        np.testing.assert_allclose(json.loads(row.target_mean),y.mean(axis=0),atol=1e-13)
    pos=native.origin_pos.to_numpy();h=native.horizon.to_numpy();end=pos+h;valid=end<len(price)
    for track in ['smoothed','returns']:
        take=valid & native.track.eq(track).to_numpy()
        actual=data.smoothed.to_numpy()[end[take]] if track=='smoothed' else signal.to_numpy()[end[take]]-signal.to_numpy()[pos[take]]
        np.testing.assert_allclose(native.loc[take,'actual'],actual,atol=1e-12)
    scored=forecasts[forecasts.status=='scored'];t=scored.origin_pos.to_numpy(dtype=int);h=scored.horizon.to_numpy(dtype=int)
    assert np.all(h<=4) and np.all(scored.forecast_steps==h+1)
    assert np.all(market.index[t+1+h]==pd.DatetimeIndex(scored.label_end))
    assert np.all(market.index[t+1]==pd.DatetimeIndex(scored.target_start))
    for track in ['smoothed','returns']:
        part=scored[scored.track==track];p=part.origin_pos.to_numpy(dtype=int);hs=part.horizon.to_numpy(dtype=int)
        x=data.smoothed.to_numpy() if track=='smoothed' else signal.to_numpy()
        np.testing.assert_allclose(part.actual,x[p+1+hs]-x[p+1],atol=1e-12)
    # Reconstruct every new distribution from complete native forecast-error vectors.
    for (track,model),part in native.groupby(['track','model']):
        point=part.pivot(index='origin_pos',columns='horizon',values='prediction').sort_index()
        errors=part.pivot(index='origin_pos',columns='horizon',values='error').sort_index()
        name='TS_Ridge_empirical' if model=='Ridge_10' else 'TS_null_empirical'
        for row in audit[(audit.track==track)&(audit.model==name)].itertuples():
            past=errors.loc[errors.index+5<=row.origin_pos].dropna().tail(126)
            assert past.index[0]==row.calibration_first_origin_pos and past.index[-1]==row.calibration_last_origin_pos
            joint=point.loc[row.origin_pos].to_numpy()-past.to_numpy()
            delayed=joint[:,1:]-joint[:,:1];ordered=np.sort(delayed,axis=0)
            f=forecasts[(forecasts.track==track)&(forecasts.model==name)&(forecasts.origin_pos==row.origin_pos)].sort_values('horizon')
            np.testing.assert_allclose(f['mean'],delayed.mean(axis=0),atol=1e-12)
            np.testing.assert_allclose(f.q10,ordered[12],atol=1e-12)
            np.testing.assert_allclose(f.q90,ordered[113],atol=1e-12)
            if f.status.eq('scored').all():
                truth=f.actual.to_numpy();weights=(2*np.arange(1,127)-127)[:,None]
                crps=abs(delayed-truth).mean(axis=0)-np.sum(weights*ordered,axis=0)/126**2
                np.testing.assert_allclose(f.CRPS,crps,atol=1e-12)
    # Baseline reproduction is checked from raw fields, not only a saved 'passed' flag.
    old=pd.read_csv(ref/'nb9/forecasts.csv',parse_dates=['date'])
    replay=pd.read_csv(folder/'legacy_replayed_distributions.csv',parse_dates=['date'])
    paired=replay[replay.track=='returns'].merge(old,on=['date','model','horizon'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(paired)==9810
    for field in ['mean','q10','median','q90','p_up','actual']:np.testing.assert_allclose(paired[field+'_new'],paired[field+'_old'],atol=1e-12)
    np.testing.assert_allclose(paired.CRPS,paired.crps,atol=1e-12)
    metrics=pd.read_csv(folder/'strategy_metrics.csv',index_col=0)
    baseline=pd.read_csv(ref/'nb10/strategy_metrics.csv',index_col=0)
    np.testing.assert_allclose(metrics.loc[baseline.index,baseline.columns],baseline,atol=1e-9,equal_nan=True)
    c=protocol['decision_parameters']['cost_bps']/1e4;capital=protocol['decision_parameters']['initial_capital']
    for model in [*protocol['comparison_models'],'buy_hold','cash']:
        daily=pd.read_csv(folder/f'{model}_daily_ledger.csv',index_col=0,parse_dates=True)
        trades=pd.read_csv(folder/f'{model}_trades.csv',parse_dates=['signal_date','entry_date','exit_date'])
        assert np.isclose((1+daily['return']).prod(),daily.equity.iloc[-1]/capital)
        assert np.isclose(daily.equity.iloc[-1]-capital,trades.pnl.sum(),atol=1e-7)
        np.testing.assert_allclose(daily.equity,daily.cash+daily.units*price.reindex(daily.index),atol=1e-7)
        if len(trades):
            np.testing.assert_allclose(trades.net_return,(trades.exit_price/trades.entry_price)*(1-c)/(1+c)-1,atol=1e-12)
            np.testing.assert_allclose(trades.entry_fee,trades.budget*c/(1+c),atol=1e-10)
            assert np.isclose(trades.entry_fee.sum()+trades.exit_fee.sum(),daily.fee.sum())
            assert (trades.entry_date>trades.signal_date).all()
        if model in baseline.index:
            old_daily=pd.read_csv(ref/f'nb10/{model}_daily_ledger.csv',index_col=0,parse_dates=True)
            assert old_daily.index.equals(daily.index)
            np.testing.assert_allclose(daily,old_daily,atol=1e-7,rtol=1e-10)
        if model in protocol['comparison_models']:
            plan=pd.read_csv(folder/f'{model}_decision_schedule.csv',parse_dates=['date'])
            t=plan.signal_pos.to_numpy();assert np.all(plan.entry_pos==t+1) and np.all(plan.exit_pos==t+4)
            vol=np.log(price).diff().rolling(20).std()*np.sqrt(252)
            np.testing.assert_allclose(plan.vol_ann,vol.reindex(plan.date),atol=1e-12)
            threshold=np.log((1+c)/(1-c))+5/1e4
            eligible=plan.supported.astype(bool)&(plan['mean']>threshold)
            weight=np.minimum.reduce([np.ones(len(plan)),.1/plan.vol_ann,.01/np.maximum(-np.expm1(plan.q10),1e-6)])*eligible
            np.testing.assert_allclose(plan.weight,weight,atol=1e-12)
    latest=pd.read_csv(folder/'research_only_latest_handoff.csv')
    assert latest.execution_status.eq('research_only_no_order').all() and latest.status.eq('immature').all()
    assert latest.horizon.between(1,4).all() and latest.forecast_steps.le(5).all()
    costs=pd.read_csv(folder/'frozen_decision_cost_sensitivity.csv')
    for model,part in costs.groupby('strategy'):assert (np.diff(part.sort_values('cost_bps').total_return)<=1e-12).all()
    print('TS10: exact targets, joint-sample adapter, mature calibration, legacy replay and cash/share accounting passed.')
