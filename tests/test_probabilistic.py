"""Independent score formulas, calibration timing, empirical CDFs and prefix isolation."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from tslab.probabilistic import (pinball,interval_score,weighted_interval_score,empirical_crps,
    gaussian_crps,rank_radius,distribution_intervals,ProbabilitySpec,point_ledger,
    probabilistic_replay,calibration_slice)
from tslab.supervised import FeatureSpec


def test_scores_against_elementary_definitions():
    assert pinball(3,1,.8)==pytest.approx(1.6)
    assert pinball(0,1,.8)==pytest.approx(.2)
    assert interval_score(2,0,1,.8)==pytest.approx(11)
    assert interval_score(.5,0,1,.8)==1
    got=weighted_interval_score(2,.5,np.array([0.]),np.array([1.]),[.8])
    assert got==pytest.approx((.5*1.5+.1*11)/1.5)
    assert gaussian_crps(0,1,0)==pytest.approx((np.sqrt(2)-1)/np.sqrt(np.pi))
    samples=np.array([-2.,0.,1.,4.])
    brute=np.mean(abs(samples-.3))-.5*np.mean(abs(samples[:,None]-samples[None,:]))
    assert empirical_crps(samples,.3)==pytest.approx(brute)
    assert empirical_crps(np.array([2.]),0.)==2


def test_rank_order_and_small_sample_infinity():
    a=np.arange(1.,10.)
    assert rank_radius(a,.8)==8
    assert np.isinf(rank_radius(a,.95))
    # ceil((126+1)*.95)=121, not the interpolated numpy percentile.
    assert rank_radius(np.arange(126.),.95)==120


def test_empirical_distribution_order_crps_and_joint_band():
    rng=np.random.default_rng(9);e=rng.normal(size=(126,5));s=np.ones_like(e)
    results,dist=distribution_intervals(e,np.arange(5.),s,np.ones(5))
    for method,values in results.items():
        assert (values['lower']<=values['upper']).all()
        assert (np.diff(values['lower'],axis=0)<=0).all()
        assert (np.diff(values['upper'],axis=0)>=0).all()
    np.testing.assert_array_equal(dist['samples'],e+np.arange(5.))
    # Max-score bands contain their marginal scaled-rank counterparts at same level.
    assert (results['joint_max_scaled']['lower']<=results['rank_scaled']['lower']).all()
    assert (results['joint_max_scaled']['upper']>=results['rank_scaled']['upper']).all()
    # All-origin scale factor cancels in scaled score calibration.
    doubled,_=distribution_intervals(e,np.arange(5.),s*2,np.ones(5)*2)
    np.testing.assert_array_equal(doubled['rank_scaled']['lower'],results['rank_scaled']['lower'])


def small_series():
    return pd.Series(4+np.random.default_rng(4).normal(0,.01,325).cumsum(),
                     index=pd.bdate_range('2020-01-01',periods=325))


def test_calibration_uses_issued_errors_and_full_label_maturity():
    spec=ProbabilitySpec(forecast_start=180,evaluation_start=230,evaluation_step=20,
        max_samples=80,calibration_window=40,scale_window=30)
    x=small_series();ledger,audit,_=point_ledger(x,spec,FeatureSpec(warmup=60))
    part=ledger[(ledger.track=='returns')&(ledger.model=='Ridge_10')]
    e,s,idx=calibration_slice(part,260,40)
    assert idx[-1]==255 and idx[0]==216
    poison=part.copy();poison.loc[poison.origin_pos>255,'error']=1e6
    ee,ss,ii=calibration_slice(poison,260,40)
    np.testing.assert_array_equal(e,ee);np.testing.assert_array_equal(s,ss);np.testing.assert_array_equal(idx,ii)
    run=probabilistic_replay(ledger,spec)
    assert (run['calibration_audit'].calibration_last_label_pos<=run['calibration_audit'].origin_pos).all()
    assert run['intervals'].query('origin_pos==324').status.eq('immature').all()
    assert (audit.max_training_label<=audit.origin_pos).all()
    assert run['joint_coverage'].n.nunique()==1
    q=run['quantiles'].pivot(index=['origin','track','model','method','horizon'],columns='tau',values='quantile')
    assert (np.diff(q.to_numpy(),axis=1)>=0).all()


def test_full_forecast_and_interval_prefix_poison():
    spec=ProbabilitySpec(forecast_start=180,evaluation_start=230,evaluation_step=20,
        max_samples=80,calibration_window=40,scale_window=30)
    x=small_series();poison=x.copy();poison.iloc[251:]+=200
    a,_,_=point_ledger(x,spec,FeatureSpec(warmup=60))
    b,_,_=point_ledger(poison,spec,FeatureSpec(warmup=60))
    cols=['origin','track','model','horizon','prediction','scale']
    pd.testing.assert_frame_equal(a[a.origin_pos<=250][cols],b[b.origin_pos<=250][cols])
    aa=probabilistic_replay(a,spec)['intervals'];bb=probabilistic_replay(b,spec)['intervals']
    cols=['origin','track','model','method','horizon','level','lower','upper','median','calibration_last_label_pos']
    pd.testing.assert_frame_equal(aa[aa.origin_pos<=250][cols],bb[bb.origin_pos<=250][cols])


@pytest.mark.parametrize('args',[{'calibration_window':2},{'max_samples':39},{'evaluation_step':0}])
def test_probability_budget_rejection(args):
    with pytest.raises(ValueError):ProbabilitySpec(**args)


def test_iid_rank_control_and_shift_counterexample():
    from tslab.probabilistic import simulation_controls
    a=simulation_controls(seed=7,repetitions=20,calibration_n=126,test_n=500)
    b=simulation_controls(seed=7,repetitions=20,calibration_n=126,test_n=500)
    pd.testing.assert_frame_equal(a,b)
    s=a.query('level==.95').groupby(['case','method']).marginal_coverage.mean()
    assert .91<s.loc[('IID','rank_absolute')]<.99
    assert s.loc[('Abrupt 3x scale shift','rank_absolute')]<.65
    assert s.loc[('Abrupt 3x scale shift','oracle_scaled_rank')]==pytest.approx(s.loc[('IID','rank_absolute')])


@pytest.mark.parametrize('levels',[(.8,.5),(.8,.8),(1.,)])
def test_interval_probability_order_rejection(levels):
    with pytest.raises(ValueError):
        distribution_intervals(np.ones((40,5)),np.zeros(5),np.ones((40,5)),np.ones(5),levels)


def test_simulation_budget_rejection():
    from tslab.probabilistic import simulation_controls
    with pytest.raises(ValueError):simulation_controls(repetitions=0)
