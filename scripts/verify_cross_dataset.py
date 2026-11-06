"""Audit saved TS09 labels, selections, failure policy, seed means and checkpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from tslab.cross_dataset import load_cross_inputs,macro_scores,synchronized_panel_intervals
from tslab.supervised import SupervisedData,FeatureSpec,chronological_holdout
from tslab.evaluation import scorecard
from tslab.neural import NeuralFit
from tslab.data import digest


def verify_cross_dataset(root,folder,manifest):
    protocol=json.loads((root/'TS09_PROTOCOL.json').read_text())
    assert manifest['config']==protocol
    assert digest(root/'TS09_PROTOCOL.json')==manifest['source']['protocol_sha256']
    stage=json.loads((folder/'completed_model_stage.json').read_text())
    assert stage['protocol_sha256']==manifest['source']['protocol_sha256']
    from benchmark_cache import sources_compatible
    assert sources_compatible(root,stage['source_sha256'])
    for name,checksum in stage['files_sha256'].items():assert digest(folder/name)==checksum
    signals,catalogue=load_cross_inputs(root/'inputs/cross_dataset')
    assert {m['dataset_id']:m['sha256'] for m in catalogue}==protocol['input_sha256']
    read=lambda n:pd.read_csv(folder/(n+'.csv'))
    forecast=read('historical_forecasts');inner=read('inner_forecasts');selections=read('selections')
    scores=read('scores');members=read('seed_forecasts')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    stock_ids=[m['dataset_id'] for m in catalogue if m['group']=='stocks']
    for identifier,x in signals.items():
        data=SupervisedData(x,FeatureSpec(**protocol['features']))
        f=forecast[forecast.dataset_id==identifier].copy()
        pos=f.origin_pos.to_numpy(dtype=int);h=f.horizon.to_numpy(dtype=int);end=pos+h
        valid=end<len(x)
        assert np.all((pos+5<len(x)) | (pos==len(x)-1))
        assert f.loc[~valid,'status'].eq('immature').all()
        for track in ['smoothed','returns']:
            mask=valid & f.track.eq(track).to_numpy()
            y=data.smoothed.to_numpy()[end[mask]] if track=='smoothed' else x.to_numpy()[end[mask]]-x.to_numpy()[pos[mask]]
            np.testing.assert_allclose(f.loc[mask,'actual'],y,atol=1e-12)
        pd.testing.assert_series_equal(pd.to_datetime(f.origin),pd.Series(x.index[pos],index=f.index,name='origin'),check_dtype=False)
        assert (pd.to_datetime(f.training_cutoff)<=pd.to_datetime(f.origin)).all()
        assert (pd.to_datetime(f.data_available_through)<=pd.to_datetime(f.origin)).all()
        expected=scorecard(f).sort_values(['track','horizon','model']).reset_index(drop=True)
        saved=scores[scores.dataset_id==identifier].sort_values(['track','horizon','model']).reset_index(drop=True)
        np.testing.assert_allclose(saved[['n','MAE','RMSE','MAE_skill','MSE_skill']],expected[['n','MAE','RMSE','MAE_skill','MSE_skill']],atol=1e-11,equal_nan=True)
        choices=selections[selections.dataset_id==identifier]
        assert (pd.to_datetime(choices.inner_last_label)<=pd.to_datetime(choices.selection_cutoff)).all()
        assert (pd.to_datetime(choices.selection_cutoff)<pd.to_datetime(choices.outer_first_origin)).all()
        lookup=f.set_index(['origin','track','horizon','model']).prediction
        choice=choices.set_index(['block','track']).chosen_model
        for row in f[f.model=='selected_pipeline'].itertuples():
            assert row.chosen_model==choice.loc[(row.block,row.track)]
            assert row.prediction==lookup.loc[(row.origin,row.track,row.horizon,row.chosen_model)]
        cutoffs=choices.groupby('block').selection_cutoff.first()
        i=inner[inner.dataset_id==identifier]
        assert (pd.to_datetime(i.available_at)<=pd.to_datetime(i.block.map(cutoffs))).all()
        # Fallback predictions are explicit null values, not dropped outer rows.
        for track,base in [('smoothed','filter_nochange'),('returns','zero')]:
            null=f[(f.track==track)&(f.model==base)].set_index(['origin','horizon']).prediction
            for row in f[(f.track==track)&(f.fit_status=='fallback')].itertuples():
                assert row.prediction==null.loc[(row.origin,row.horizon)]
        cache={}
        for name in ['tabular_audit','neural_audit']:
            audit=read(name);audit=audit[audit.dataset_id==identifier]
            assert (audit.max_training_label<=audit.origin_pos).all()
            for row in audit[audit.fit_status=='ok'].itertuples():
                if str(row.seed)=='ensemble':continue
                key=(row.origin_pos,row.track)
                if key not in cache:cache[key]=data.training(row.origin_pos,row.track,252)
                X,y,origins=cache[key]
                assert row.n_train==len(X) and row.max_training_label==origins[-1]+5
                mean=json.loads(row.feature_mean)
                if name=='tabular_audit':np.testing.assert_allclose(mean,X.reshape(len(X),-1).mean(axis=0),atol=1e-13)
                else:
                    tr,va=chronological_holdout(origins)
                    assert row.optimization_last_label==origins[tr[-1]]+5<row.validation_first_origin==origins[va[0]]
                    np.testing.assert_allclose(mean,X.mean(axis=(0,1)),atol=1e-13)
                    np.testing.assert_allclose(json.loads(row.stopping_feature_mean),X[tr].mean(axis=(0,1)),atol=1e-13)
        classic=read('classical_audit');classic=classic[classic.dataset_id==identifier]
        assert (classic.bridge_last_label_pos<=classic.origin_pos).all()
        member=members[members.dataset_id==identifier]
        means=member.groupby(['origin','track','horizon','model']).prediction.mean()
        assert member.groupby(['origin','track','horizon','model']).seed.nunique().eq(3).all()
        for row in pd.concat([f,i])[lambda z:z.model.isin(['GRU','NBEATS_demo']) & z.fit_status.eq('ok')].itertuples():
            assert abs(row.prediction-means.loc[(row.origin,row.track,row.horizon,row.model)])<1e-11
        for checkpoint in (folder/'checkpoints'/identifier).glob('*.pt'):
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            fit=NeuralFit.restore(state);query=data.features[state['origin_pos']]
            np.testing.assert_array_equal(query,state['sequence'].numpy())
            prediction=fit.predict(query)
            np.testing.assert_array_equal(prediction,state['residual_prediction'].numpy())
            np.testing.assert_array_equal(data.decode(prediction,state['origin_pos'],state['track']),state['decoded_prediction'].numpy())
    for name,checksum in manifest['checkpoint_sha256'].items():assert digest(folder/name)==checksum
    expected=macro_scores(scores,catalogue).sort_values(['group','track','horizon','model']).reset_index(drop=True)
    saved=read('macro_scorecard').sort_values(['group','track','horizon','model']).reset_index(drop=True)
    cols=['datasets','mean_MAE_skill','mean_MSE_skill','median_MSE_skill','min_MSE_skill','max_MSE_skill','positive_datasets']
    np.testing.assert_allclose(saved[cols],expected[cols],atol=1e-11,equal_nan=True)
    # Recheck aligned panel bootstrap against saved data; algebra is unit-tested independently.
    expected=synchronized_panel_intervals(forecast,stock_ids)
    np.testing.assert_allclose(read('synchronized_stock_panel_uncertainty').select_dtypes('number'),expected.select_dtypes('number'),atol=1e-12)
    audit=read('independent_evaluation_audit')
    assert audit[audit.criterion=='Pristine independent asset/period holdout'].status.eq('not established').all()
    assert protocol['independence']['prospective_evaluation_completed'] is False
    print('TS09: dataset/target isolation, timing, paired scores, scalers, seeds, checkpoints and independence audit passed.')
