"""Architecture invariants, refit reproducibility, checkpoint and prefix isolation."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from tslab.advanced import advanced_models,LinearAdapter,ResidualStack,PatchTransformer
from tslab.neural import ForecastNet,NeuralFit,fit_neural,deterministic_cpu
from tslab.learned_evaluation import LearnedRunner
from tslab.supervised import SupervisedData


@pytest.mark.parametrize('spec',advanced_models(),ids=lambda s:s.name)
def test_advanced_fit_checkpoint_and_future_isolation(spec):
    x=pd.Series(4+np.cumsum(np.random.default_rng(30).normal(0,.01,340)),
                index=pd.bdate_range('2020-01-01',periods=340))
    altered=x.copy();altered.iloc[281:]+=500
    spec=replace(spec,max_epochs=3,patience=2)
    data=SupervisedData(x)
    X,y,origins=data.training(280,'returns',80)
    fit,audit=fit_neural(X,y,origins,spec,seed=17)
    repeat,_=fit_neural(X,y,origins,spec,seed=17)
    query=data.features[280]
    np.testing.assert_array_equal(fit.predict(query),repeat.predict(query))
    np.testing.assert_array_equal(fit.predict(query),NeuralFit.restore(fit.checkpoint()).predict(query))
    assert np.isfinite(fit.predict(query)).all()
    assert audit['optimization_last_label']<audit['validation_first_origin']
    kwargs=dict(models=[spec],mode='neural',seeds=(17,),max_samples=80)
    pd.testing.assert_frame_equal(LearnedRunner(x,**kwargs).predictions(280),
                                  LearnedRunner(altered,**kwargs).predictions(280))


def test_linear_decomposition_identity_and_centering_translation():
    torch.manual_seed(3)
    net=LinearAdapter(20,2,True)
    # Equal temporal maps make trend + residual exactly the same linear map plus two biases.
    net.trend.load_state_dict(net.temporal.state_dict())
    x=torch.randn(4,20,2)
    expected=net.readout((net.temporal(x.transpose(1,2))+net.temporal.bias).transpose(1,2)).squeeze(-1)
    torch.testing.assert_close(net(x),expected,atol=3e-7,rtol=1e-6)
    centered=LinearAdapter(20,2,False)
    shift=torch.tensor([2.,-3.])
    delta=(shift*centered.readout.weight[0]).sum()
    torch.testing.assert_close(centered(x+shift)-centered(x),delta.expand(4,5),atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('multirate',[False,True])
def test_residual_forecast_is_sum_and_backcast_affects_later_blocks(multirate):
    torch.manual_seed(8)
    net=ResidualStack(20,2,12,multirate)
    x=torch.randn(3,20,2)
    residual=x.flatten(1);expected=torch.zeros(3,5)
    for block in net.blocks:
        backcast,forecast=block(residual)
        residual=residual-backcast;expected+=forecast
    torch.testing.assert_close(net(x),expected,atol=0,rtol=0)
    before=net(x).detach().clone()
    with torch.no_grad():net.blocks[0].backcast.bias.add_(10.)
    assert not torch.equal(before,net(x))


def test_patch_positions_and_batch_independence():
    deterministic_cpu(9)
    net=PatchTransformer(20,2,12).eval()
    x=torch.randn(3,20,2)
    with torch.no_grad():
        torch.testing.assert_close(net(x)[0],net(x[:1])[0],atol=2e-7,rtol=1e-6)
        assert not torch.allclose(net(x),net(x.flip(1)))
    with pytest.raises(ValueError):PatchTransformer(19,2,12)
