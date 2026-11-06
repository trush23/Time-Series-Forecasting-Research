"""Label timing, recursive dynamics, train-only scales and neural reproducibility."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from tslab.supervised import FeatureSpec,SupervisedData,ema_anchor,chronological_holdout,recursive_rollout
from tslab.smoothing import CausalSmoother,FilterSpec
from tslab.tabular import TabularSpec,fit_tabular,tabular_models
from tslab.neural import NeuralSpec,ForecastNet,CausalBlock,fit_neural,NeuralFit
from tslab.learned_evaluation import LearnedRunner,nested_learned


def series(n=450):
    return pd.Series(4+np.random.default_rng(1731).normal(0,.01,n).cumsum(),
                     index=pd.bdate_range('2020-01-01',periods=n))


def test_features_maturity_anchor_and_prefix_poison():
    x=series()
    data=SupervisedData(x)
    X,y,origins=data.training(300,'returns')
    assert np.max(origins+5)<=300
    assert len(origins)<=252 and np.isfinite(X).all()
    poison=x.copy();poison.iloc[301:]+=100
    other=SupervisedData(poison)
    for track in ['smoothed','returns']:
        a,b,idx=data.training(300,track)
        aa,bb,idx2=other.training(300,track)
        np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)
        np.testing.assert_array_equal(idx,idx2)
    engine=CausalSmoother(FilterSpec())
    smooth=engine.update_many(x.iloc[:301])
    np.testing.assert_allclose(ema_anchor(smooth[-1]-smooth[-2])+smooth[-1],
                               engine.project(np.repeat(x.iloc[300],5)),atol=1e-12)
    # The one-step residual target exactly equals alpha * actual next raw return.
    np.testing.assert_allclose(data.labels['smoothed'][origins,0],
                               .2*data.labels['returns'][origins,0],atol=1e-12)


def test_chronological_purge_and_recursive_null():
    origins=np.arange(100,200)
    train,val=chronological_holdout(origins)
    assert origins[train[-1]]+5<origins[val[0]]
    sequence=np.random.default_rng(4).normal(0,.001,(20,2))
    np.testing.assert_allclose(recursive_rollout(lambda _:0.,sequence,'smoothed'),0,atol=1e-16)
    np.testing.assert_equal(recursive_rollout(lambda _:0.,sequence,'returns'),np.zeros(5))
    np.testing.assert_allclose(recursive_rollout(lambda _: .001,sequence,'returns'),np.arange(1,6)*.001)


def test_ridge_independent_planted_linear_mapping_and_scales():
    rng=np.random.default_rng(5)
    X=rng.normal(size=(120,20,2))
    beta=rng.normal(size=(40,5))
    y=X.reshape(120,-1)@beta
    fitted,audit=fit_tabular(X,y,TabularSpec('linear','ridge',1e-8))
    query=rng.normal(size=(20,2))
    np.testing.assert_allclose(fitted.predict(query),query.reshape(-1)@beta,atol=1e-6)
    np.testing.assert_allclose(audit['feature_mean'],X.reshape(120,-1).mean(axis=0))


@pytest.mark.parametrize('model',tabular_models(),ids=lambda m:m.name)
def test_tabular_shapes_and_finite_predictions(model):
    data=SupervisedData(series())
    X,y,idx=data.training(300,'returns',100)
    fit,_=fit_tabular(X,y,model)
    assert fit.predict(data.features[300]).shape==(5,)
    assert np.isfinite(fit.predict(data.features[300])).all()


@pytest.mark.parametrize('family',['mlp','lstm','gru','tcn'])
def test_neural_determinism_checkpoint_and_train_only_stopping_scales(family):
    data=SupervisedData(series(320),FeatureSpec(lookback=8,warmup=60))
    X,y,origins=data.training(250,'returns',80)
    spec=NeuralSpec(family,family,hidden=4,max_epochs=3,patience=2)
    fitted,audit=fit_neural(X,y,origins,spec,seed=19)
    repeat,_=fit_neural(X,y,origins,spec,seed=19)
    pred=fitted.predict(data.features[250])
    np.testing.assert_array_equal(pred,repeat.predict(data.features[250]))
    restored=NeuralFit.restore(fitted.checkpoint())
    np.testing.assert_array_equal(pred,restored.predict(data.features[250]))
    assert audit['optimization_last_label']<audit['validation_first_origin']
    assert audit['last_training_label']<=250
    train,val=chronological_holdout(origins,spec.validation_fraction)
    np.testing.assert_allclose(audit['stopping_feature_mean'],X[train].mean(axis=(0,1)))
    np.testing.assert_allclose(audit['feature_mean'],X.mean(axis=(0,1)))
    assert audit['chosen_epochs']==min(audit['trace'],key=lambda row:row['validation_MSE_standardized'])['epoch']


def test_tcn_internal_temporal_causality():
    torch.manual_seed(1)
    block=CausalBlock(2,4,2)
    x=torch.randn(3,2,20)
    poison=x.clone();poison[:,:,11:]+=100
    torch.testing.assert_close(block(x)[:,:,:11],block(poison)[:,:,:11],rtol=0,atol=0)


def test_learned_predictions_poison_and_failure_visibility(monkeypatch):
    x=series(360)
    models=[TabularSpec('Ridge','ridge',10.)]
    a=LearnedRunner(x,models)
    poison=x.copy();poison.iloc[301:]+=100
    b=LearnedRunner(poison,models)
    pd.testing.assert_frame_equal(a.predictions(300),b.predictions(300))
    def failed(*args,**kwargs):raise RuntimeError('forced fit failure')
    monkeypatch.setattr('tslab.learned_evaluation.fit_tabular',failed)
    run=nested_learned(x,models,outer_starts=(280,),outer_step=30,inner_count=2,inner_step=20)
    assert set(run['selections'].chosen_model)=={'filter_nochange','zero'}
    assert run['fit_audit'].fit_status.eq('fallback').all()
    assert (run['fit_audit'].max_training_label<=run['fit_audit'].origin_pos).all()


def test_neural_runner_future_poison_and_seed_ensemble():
    x=series(320)
    models=[NeuralSpec('GRU','gru',hidden=4,max_epochs=2,patience=1)]
    kwargs=dict(models=models,mode='neural',max_samples=80,seeds=(1,2))
    a=LearnedRunner(x,**kwargs)
    poison=x.copy();poison.iloc[251:]+=100
    b=LearnedRunner(poison,**kwargs)
    pa,pb=a.predictions(250),b.predictions(250)
    pd.testing.assert_frame_equal(pa,pb)
    members=pd.DataFrame(a.members).groupby(['track','horizon']).prediction.mean()
    for row in pa[pa.model=='GRU'].itertuples():
        assert row.prediction==pytest.approx(members.loc[(row.track,row.horizon)],abs=1e-12)


def test_failed_neural_seed_keeps_all_members_and_uses_whole_ensemble_fallback(monkeypatch):
    from tslab import neural
    original=neural.fit_neural
    def fail_one(X,y,origins,spec,seed):
        if seed==2:raise RuntimeError('forced seed failure')
        return original(X,y,origins,spec,seed)
    monkeypatch.setattr(neural,'fit_neural',fail_one)
    runner=LearnedRunner(series(320),[NeuralSpec('GRU','gru',hidden=4,max_epochs=2,patience=1)],
                         mode='neural',max_samples=80,seeds=(1,2,3))
    records=runner.predictions(250)
    assert len(runner.members)==2*3*5
    assert records[records.model=='GRU'].fit_status.eq('fallback').all()
    for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
        np.testing.assert_equal(records[(records.track==track)&(records.model=='GRU')].prediction.to_numpy(),
             records[(records.track==track)&(records.model==baseline)].prediction.to_numpy())


@pytest.mark.parametrize('kwargs',[{'lookback':1},{'fast_alpha':0},{'slow_alpha':.3}])
def test_feature_configuration_rejection(kwargs):
    with pytest.raises(ValueError):FeatureSpec(**kwargs)
