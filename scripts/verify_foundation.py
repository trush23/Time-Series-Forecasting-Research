"""Independently reconstruct TS11 labels, adapters, scores and residual intervals."""
from pathlib import Path
from hashlib import sha256
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from tslab.data import digest
from tslab.cross_dataset import load_cross_inputs
from tslab.supervised import SupervisedData


def verify_foundation(root,folder,manifest):
    root,folder=Path(root),Path(folder)
    protocol=json.loads((root/'TS11_PROTOCOL.json').read_text())
    assert digest(root/'TS11_PROTOCOL.json')==manifest['protocol_sha256']
    assert protocol==manifest['config']
    inference=json.loads((folder/'inference_manifest.json').read_text())
    assert digest(folder/'inference_manifest.json')==manifest['inference_manifest_sha256']
    assert digest(folder/'foundation_quantiles.npz')==manifest['inference_cache_sha256']==inference['cache_sha256']
    assert digest(root/'src/tslab/foundation.py')==inference['identity']['module_sha256']
    checkpoints=json.loads((root/'inputs/foundation_models/checkpoint_manifest.json').read_text())
    assert digest(root/'inputs/foundation_models/checkpoint_manifest.json')==protocol['checkpoint_manifest_sha256']
    for checkpoint in checkpoints:
        for name,expected in checkpoint['files_sha256'].items():
            assert digest(root/'inputs/foundation_models'/checkpoint['folder']/name)==expected
    signals,catalogue=load_cross_inputs(root/'inputs/cross_dataset')
    ledger=pd.read_csv(folder/'historical_forecasts.csv',parse_dates=['origin','target_end','data_available_through'])
    assert len(ledger)==8*81*6*2*5
    assert ledger[ledger.role=='evaluation'].groupby(['dataset_id','track','model','horizon']).size().eq(40).all()
    assert ledger[ledger.role=='latest'].actual.isna().all()
    assert (ledger.origin==ledger.data_available_through).all()
    cache=np.load(folder/'foundation_quantiles.npz',allow_pickle=False)
    contexts=pd.read_csv(folder/'context_audit.csv')
    for name,signal in signals.items():
        data=SupervisedData(signal);x,s=signal.to_numpy(),data.smoothed.to_numpy()
        for track in ['smoothed','returns']:
            scored=ledger[(ledger.dataset_id==name)&(ledger.track==track)&(ledger.status=='scored')]
            t=scored.origin_pos.to_numpy();h=scored.horizon.to_numpy()
            expected=s[t+h] if track=='smoothed' else x[t+h]-x[t]
            np.testing.assert_allclose(scored.actual,expected,atol=1e-12)
            assert (pd.DatetimeIndex(scored.target_end)==signal.index[t+h]).all()
        for model,part in contexts[contexts.dataset_id==name].groupby('model'):
            part=part.sort_values('origin_pos');length=int(model.split('_L')[1])
            for k,row in enumerate(part.itertuples()):
                t=row.origin_pos;v=s[t-length+1:t+1]-s[t]
                assert sha256(v.tobytes()).hexdigest()==row.context_sha256
                assert row.context_start_pos==t-length+1 and row.context_end_pos==t
                native=cache[f'{model}__{name}'][k];future=native[4]+s[t]
                assert row.quantile_crossings==int((np.diff(native,axis=0)<0).sum())
                actual_path=(future-.8*np.r_[s[t],future[:-1]])/.2-x[t]
                for track,expected in [('smoothed',future),('returns',actual_path)]:
                    rows=ledger[(ledger.dataset_id==name)&(ledger.model==model)&(ledger.track==track)&(ledger.origin_pos==t)].sort_values('horizon')
                    np.testing.assert_allclose(rows.prediction,expected,atol=2e-12)
    fits=pd.read_csv(folder/'ridge_fit_audit.csv')
    assert len(fits)==8*81*2 and (fits.max_training_label_pos<=fits.origin_pos).all()
    scores=pd.read_csv(folder/'point_scores.csv')
    for row in scores.itertuples():
        p=ledger[(ledger.dataset_id==row.dataset_id)&(ledger.track==row.track)&(ledger.model==row.model)&(ledger.horizon==row.horizon)&(ledger.role=='evaluation')]
        error=p.prediction-p.actual
        np.testing.assert_allclose([row.MAE,row.MSE],[np.abs(error).mean(),np.square(error).mean()],atol=1e-12)
        np.testing.assert_allclose([row.MAE_skill,row.MSE_skill],[1-row.MAE/row.null_MAE,1-row.MSE/row.null_MSE],atol=1e-10)
    macro=pd.read_csv(folder/'macro_scores.csv')
    for row in macro.itertuples():
        part=scores[(scores.group==row.group)&(scores.track==row.track)&(scores.model==row.model)&(scores.horizon==row.horizon)]
        assert len(part)==row.datasets
        np.testing.assert_allclose([row.MAE_skill,row.MSE_skill],part[['MAE_skill','MSE_skill']].mean(),atol=1e-10)
    intervals=pd.read_csv(folder/'residual_intervals.csv')
    audit=pd.read_csv(folder/'calibration_audit.csv')
    assert audit.calibration_n.eq(32).all() and (audit.last_calibration_label<=audit.origin_pos).all()
    for (name,track,model),group in intervals.groupby(['dataset_id','track','model']):
        part=ledger[(ledger.dataset_id==name)&(ledger.track==track)&(ledger.model==model)]
        residuals=part.assign(error=part.actual-part.prediction).pivot(index='origin_pos',columns='horizon',values='error')
        for t,rows in group.groupby('origin_pos'):
            past=residuals.loc[residuals.index+5<=t].dropna().tail(32)
            point=part[part.origin_pos==t].sort_values('horizon').prediction.to_numpy()
            expected=point+np.quantile(past,[.1,.5,.9],axis=0,method='inverted_cdf')
            np.testing.assert_allclose(rows.sort_values('horizon')[['lower80','median','upper80']].to_numpy().T,expected,atol=2e-12)
    scored=intervals[intervals.role=='evaluation']
    assert scored.actual.notna().all()
    interval_scores=pd.read_csv(folder/'interval_scores.csv')
    for row in interval_scores.itertuples():
        part=scored[(scored.dataset_id==row.dataset_id)&(scored.track==row.track)&(scored.model==row.model)&(scored.horizon==row.horizon)]
        y,l,u=part.actual,part.lower80,part.upper80
        expected=[((l<=y)&(y<=u)).mean(),(u-l).mean(),(u-l+10*np.maximum(l-y,0)+10*np.maximum(y-u,0)).mean()]
        np.testing.assert_allclose([row.coverage80,row.width80,row.interval_score80],expected,atol=2e-11)
    control=pd.read_csv(folder/'martingale_control.csv')
    assert len(control)==10 and control.return_conditional_MSE_ratio.ge(1).all()
    print('TS11: pinned weights, 38,880 forecast rows, both tracks, adapters, scores and mature intervals verified')


if __name__=='__main__':
    import sys
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
    folder=root/'outputs/TS11_Foundation_Model_Laboratory'
    verify_foundation(root,folder,json.loads((folder/'run_manifest.json').read_text()))
