"""Target algebra, quantile dependence, pure decision inputs and exact ledger identities."""
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from tslab.integration import delayed_samples,summarize_sample,NB9Reference
from tslab.trading import TradingSpec,decision_plan,simulate_plan,trading_metrics
from tslab.probabilistic import ProbabilitySpec,point_ledger
from tslab.supervised import FeatureSpec
from tslab.handoff import bridge_ledger


def test_joint_difference_then_quantiles_not_marginal_quantile_subtraction():
    a=np.column_stack([[-2.,-1.,1.,2.],[2.,1.,-1.,-2.],np.zeros((4,3))])
    out=delayed_samples(a)
    assert out.shape==(4,4)
    np.testing.assert_array_equal(out[:,0],[4.,2.,-2.,-4.])
    assert summarize_sample(out[:,0])['q10']==-4
    assert summarize_sample(a[:,1])['q10']-summarize_sample(a[:,0])['q10']==0
    with pytest.raises(ValueError):delayed_samples(np.zeros((20,6)))


def plan_fixture():
    price=pd.Series([100.,100.,105.,103.,110.,108.,111.,112.],index=pd.bdate_range('2020-01-01',periods=8))
    p=pd.DataFrame({'date':[price.index[0]],'signal_pos':[0],'entry_pos':[1],'exit_pos':[4],'weight':[.4]})
    return price,p


@pytest.mark.parametrize('cost',[0.,5.,20.])
def test_exact_cash_share_accounting(cost):
    price,p=plan_fixture();spec=TradingSpec()
    ledger,tr=simulate_plan(p,price,spec,cost)
    c=cost/1e4;factor=1+.4*((110/100)*(1-c)/(1+c)-1)
    assert ledger.equity.iloc[-1]/spec.initial_capital==pytest.approx(factor)
    assert ledger.fee.sum()==pytest.approx(tr.entry_fee.sum()+tr.exit_fee.sum())
    assert (1+ledger['return']).prod()==pytest.approx(factor)
    p.weight=0.;cash,_=simulate_plan(p,price,spec,cost)
    assert cash.equity.eq(spec.initial_capital).all()


def test_exit_before_entry_roll_and_overlapping_position_rejection():
    price,p=plan_fixture()
    second=p.copy();second['date']=price.index[3];second['signal_pos']=3;second['entry_pos']=4;second['exit_pos']=7
    roll=pd.concat([p,second],ignore_index=True)
    ledger,tr=simulate_plan(roll,price,cost_bps=0)
    assert len(tr)==2 and ledger.units.iloc[-1]==0
    bad=roll.copy();bad.loc[1,'entry_pos']=2
    with pytest.raises(ValueError,match='Overlapping'):simulate_plan(bad,price)


def test_legacy_clock_and_actual_outcomes_do_not_set_weights():
    ref=NB9Reference(Path(__file__).resolve().parents[1]/'inputs/nb_integration')
    f=ref.forecasts[ref.forecasts.model=='analog'].copy();price=ref.market.Close
    p=decision_plan(f,price,model='analog')
    poisoned=f.copy();poisoned['actual']=1e6
    other=decision_plan(poisoned,price,model='analog')
    np.testing.assert_array_equal(p.weight,other.weight)
    assert (p.entry_pos==p.signal_pos+1).all() and (p.exit_pos==p.signal_pos+4).all()
    with pytest.raises(ValueError,match='forecast budget'):decision_plan(f,price,horizon=5)
    assert len(decision_plan(f,price,horizon=5,max_forecast_steps=6))>0
    with pytest.raises(ValueError,match='actual-return'):decision_plan(f.assign(track='smoothed'),price)


def test_legacy_neighbor_maturity_and_prefix_metric():
    ref=NB9Reference(Path(__file__).resolve().parents[1]/'inputs/nb_integration')
    t=ref.index.get_indexer([ref.forecasts.date.iloc[0]])[0]
    groups,supported,_=ref.neighbors(t);ids,w=groups['analog']
    assert (ref.positions[ids]+6<=ref.positions[t]).all()
    assert (np.diff(np.sort(ref.positions[ids]))>=5).all()
    truth=ref.targets[t,2];got=summarize_sample(ref.targets[ids,2],w,truth)
    saved=ref.forecasts[(ref.forecasts.date==ref.index[t])&(ref.forecasts.horizon==3)&(ref.forecasts.model=='analog')].iloc[0]
    assert got['mean']==pytest.approx(saved['mean'],abs=1e-12)
    assert got['CRPS']==pytest.approx(saved.crps,abs=1e-12)


def test_prequential_delayed_bridge_poisoned_future_errors():
    x=pd.Series(4+np.random.default_rng(2).normal(0,.01,320).cumsum(),index=pd.bdate_range('2020-01-01',periods=320))
    spec=ProbabilitySpec(forecast_start=180,evaluation_start=230,calibration_window=40,max_samples=80,scale_window=30)
    ledger,_,_=point_ledger(x,spec,FeatureSpec(warmup=60));support=pd.Series(True,index=x.index)
    a,audit,_=bridge_ledger(ledger,[250],x.index,support,40)
    poison=ledger.copy();poison.loc[poison.origin_pos+5>250,'error']=1e6
    b,_,_=bridge_ledger(poison,[250],x.index,support,40)
    cols=['track','horizon','model','mean','q10','q90']
    pd.testing.assert_frame_equal(a[a.origin_pos==250][cols].reset_index(drop=True),b[b.origin_pos==250][cols].reset_index(drop=True))
    assert (audit.calibration_last_label_pos<=audit.origin_pos).all()
    for row in a[(a.origin_pos==250)&(a.track=='returns')].itertuples():
        assert row.actual==pytest.approx(x.iloc[251+row.horizon]-x.iloc[251])


@pytest.mark.parametrize('kw',[{'cost_bps':10000},{'max_weight':2},{'leverage':True}])
def test_trading_spec_rejects_invalid_settings(kw):
    with pytest.raises(ValueError):TradingSpec(**kw)
