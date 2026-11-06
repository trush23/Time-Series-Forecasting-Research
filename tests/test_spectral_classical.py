"""Independent spectral identities, parameter recovery, and selection causality."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.ar_model import AutoReg
from tslab.spectral import SpectralSpec, SpectralEndpoint, fft_decompose, spectral_replay, harmonic_forecast
from tslab.classical import ModelSpec, default_models, native_forecast, fit_return_bridge, bridge_forecast
from tslab.prequential import OriginRunner, nested_benchmark, choose_models
from tslab.smoothing import FilterSpec


def test_fft_exact_reconstruction_and_bandpass_dc():
    t=np.arange(128)
    x=4+.1*np.sin(2*np.pi*t/16)+.03*np.cos(2*np.pi*t/4)
    fit=fft_decompose(x,low=1/20,high=1/12,detrend=False)
    np.testing.assert_allclose(fit['trend']+fit['band']+fit['remainder'],x,atol=1e-14)
    np.testing.assert_allclose(fit['band'],.1*np.sin(2*np.pi*t/16),atol=1e-14)
    assert fit['band'].mean()==pytest.approx(0,abs=1e-14)


def test_fft_constant_and_ramp():
    t=np.arange(128)
    for x in [np.full(128,7.),4+.01*t]:
        fit=fft_decompose(x,detrend=True)
        np.testing.assert_allclose(fit['trend'],x,atol=1e-12)
        np.testing.assert_allclose(fit['band'],0,atol=1e-12)


def test_fft_prefix_poison_state_projection():
    x=np.random.default_rng(4).normal(size=220)
    spec=SpectralSpec(window=64)
    full=spectral_replay(x,spec)
    prefix=spectral_replay(x[:150],spec)
    poison=x.copy(); poison[150:]+=100
    contaminated=spectral_replay(poison,spec)
    for k in full:
        np.testing.assert_allclose(full[k][:150],prefix[k],equal_nan=True)
        np.testing.assert_allclose(full[k][:150],contaminated[k][:150],equal_nan=True)
    engine=SpectralEndpoint(spec)
    for value in x[:150]: engine.update(value)
    state=engine.checkpoint()
    engine.project([0]*5)
    assert engine.checkpoint()==state
    restored=SpectralEndpoint.restore(state)
    tail=[restored.update(v)['reconstructed'] for v in x[150:]]
    np.testing.assert_allclose(tail,full['reconstructed'][150:])


def test_harmonic_forecast_known_frequency_and_trend():
    t=np.arange(192)
    # An integer number of periods with no trend gives an exact DFT-bin test.
    x=4+.1*np.cos(2*np.pi*t/16)
    forecast,audit=harmonic_forecast(x,window=128,harmonics=1,low=.04,high=.1)
    np.testing.assert_allclose(forecast,4+.1*np.cos(2*np.pi*np.arange(192,197)/16),atol=1e-12)
    assert audit['frequencies']==pytest.approx([1/16])
    ramp=4+.001*t
    forecast,_=harmonic_forecast(ramp,window=128,harmonics=0)
    np.testing.assert_allclose(forecast,4+.001*np.arange(192,197),atol=1e-12)


@pytest.mark.parametrize('spec',default_models(),ids=lambda s:s.name)
def test_native_constant(spec):
    forecast,diagnostic=native_forecast(np.full(100,4.),spec)
    np.testing.assert_equal(forecast,np.full(5,4.))
    assert diagnostic['constant_path']


def test_ar_independent_reference():
    rng=np.random.default_rng(1731)
    d=np.empty(600); d[0]=0
    for i in range(1,len(d)): d[i]=.75*d[i-1]+rng.normal(0,.01)
    y=4+np.r_[0,d].cumsum()
    forecast,_=native_forecast(y,ModelSpec('AR1','ar_diff'),train_window=600)
    reference=AutoReg(np.diff(y[-600:]),lags=1,trend='c',old_names=False).fit()
    expected=y[-1]+np.cumsum(reference.predict(start=599,end=603))
    np.testing.assert_allclose(forecast,expected,atol=1e-12)
    assert reference.params[1]==pytest.approx(.75,abs=.12)


def test_bridge_maturity_and_known_coefficients():
    rng=np.random.default_rng(1)
    s=np.cumsum(rng.normal(size=200))
    x=np.zeros(200)
    # Next actual return depends on current smoothed slope; exact constructed labels.
    for t in range(1,199): x[t+1]=x[t]+.01+.2*(s[t]-s[t-1])
    bridge=fit_return_bridge(x,s)
    assert bridge['beta']==pytest.approx(.2)
    assert bridge['intercept']==pytest.approx(.01)
    assert bridge['last_label_pos']==199 and bridge['last_feature_pos']==198
    path=s[-1]+np.arange(1,6)*.3
    forecast=bridge_forecast(path,s,bridge)
    expected=np.cumsum(.01+.2*np.r_[s[-1]-s[-2],np.repeat(.3,4)])
    np.testing.assert_allclose(forecast,expected)


def make_series(n=330):
    return pd.Series(4+np.random.default_rng(41).normal(0,.01,n).cumsum(),
                     index=pd.bdate_range('2020-01-01',periods=n))


def test_native_prefix_forecasts_future_poison_and_fallback():
    x=make_series()
    candidates=[ModelSpec('AR1','ar_diff'),ModelSpec('too_many_lags','ar_diff',lags=200)]
    before=OriginRunner(x,candidates,train_window=100)
    poison=x.copy(); poison.iloc[250:]+=50
    after=OriginRunner(poison,candidates,train_window=100)
    a,b=before.predictions(240),after.predictions(240)
    pd.testing.assert_frame_equal(a,b)
    bad=a[a.model=='too_many_lags']
    assert bad.fit_status.eq('fallback').all()
    assert (pd.DataFrame(before.audit).bridge_last_label_pos<=240).all()
    for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
        np.testing.assert_equal(bad[bad.track==track].prediction.to_numpy(),
             a[(a.track==track)&(a.model==baseline)].prediction.to_numpy())


def test_spectral_benchmark_future_poison_and_actual_labels():
    from tslab.spectral_evaluation import spectral_benchmark
    x=make_series(350)
    spec=SpectralSpec(window=64)
    args=dict(spec=spec,start=200,step=20,harmonic_window=64)
    a,_,audit=spectral_benchmark(x,**args)
    contaminated=x.copy(); contaminated.iloc[270:]+=100
    b,_,_=spectral_benchmark(contaminated,**args)
    cols=['origin','track','horizon','model','prediction']
    pd.testing.assert_frame_equal(a.loc[a.origin<x.index[270],cols].reset_index(drop=True),
                                  b.loc[b.origin<x.index[270],cols].reset_index(drop=True))
    ret=a[(a.track=='returns')&(a.status=='scored')]
    ends=ret.origin_pos.to_numpy()+ret.horizon.to_numpy()
    np.testing.assert_allclose(ret.actual,x.to_numpy()[ends]-x.to_numpy()[ret.origin_pos])
    assert audit.fit_status.eq('ok').all()


def test_nested_selection_cutoffs_and_future_poison():
    x=make_series(330)
    models=[ModelSpec('AR1','ar_diff'),ModelSpec('SES','ses')]
    config=dict(models=models,outer_starts=(220,280),outer_step=20,inner_count=2,
                inner_step=15,train_window=100)
    run=nested_benchmark(x,**config)
    selection=run['selections']
    assert (selection.inner_last_label<=selection.selection_cutoff).all()
    assert (selection.selection_cutoff<selection.outer_first_origin).all()
    poison=x.copy(); poison.iloc[280:]+=50
    rerun=nested_benchmark(poison,**config)
    pd.testing.assert_frame_equal(selection[selection.block==0],rerun['selections'][rerun['selections'].block==0])
    a=run['forecasts']; b=rerun['forecasts']
    cols=['origin','track','horizon','model','prediction','chosen_model']
    pd.testing.assert_frame_equal(a.loc[a.origin<x.index[280],cols].reset_index(drop=True),
                                  b.loc[b.origin<x.index[280],cols].reset_index(drop=True))
    assert a[a.origin==x.index[-1]].status.eq('immature').all()
    # Immature outcomes cannot enter model choice, even with finite poisoned values.
    inner=run['inner_forecasts'].copy()
    inner.loc[inner.index[0],'status']='immature'
    with pytest.raises(ValueError,match='immature'):
        choose_models(inner,[m.name for m in models])


@pytest.mark.parametrize('kwargs',[{'window':7},{'low':.2,'high':.1},{'high':.6},{'window':10.5}])
def test_invalid_spectral_configuration(kwargs):
    with pytest.raises(ValueError): SpectralSpec(**kwargs)
