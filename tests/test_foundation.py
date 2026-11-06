"""Independent algebra, boundary and future-label contamination checks for TS11."""
import numpy as np
import pandas as pd
import pytest
from tslab.foundation import invert_ema_path,centered_context,schedule,residual_intervals,paired_ranges


@pytest.mark.parametrize('alpha',[.05,.2,.7,1.])
def test_inverse_recovers_known_actual_path(alpha):
    raw=np.array([8.,9.,7.,12.,10.,11.,6.]);s=[7.]
    for value in raw[1:]:s.append(alpha*value+(1-alpha)*s[-1])
    predicted=invert_ema_path(np.asarray(s[2:]),s[1],s[0],alpha)
    np.testing.assert_allclose(predicted,raw[2:]-raw[1],atol=2e-14)


def test_inverse_batched_paths_and_nochange():
    current=np.array([9.,-3.]);raw=np.array([10.,-4.]);previous=(current-.2*raw)/.8
    path=current[:,None]+(raw-current)[:,None]*(1-.8**np.arange(1,6))
    np.testing.assert_allclose(invert_ema_path(path,current,previous),0,atol=2e-14)


@pytest.mark.parametrize('path,alpha',[(np.zeros(4),.2),(np.full(5,np.nan),.2),(np.zeros(5),0),(np.zeros(5),1.1)])
def test_invalid_inverse_rejected(path,alpha):
    with pytest.raises(ValueError):invert_ema_path(path,1,1,alpha)


def test_context_is_prefix_only_and_translation_invariant():
    values=np.arange(100,dtype=float);changed=values.copy();changed[61:]=np.nan
    np.testing.assert_array_equal(centered_context(values,60,32),centered_context(changed,60,32))
    np.testing.assert_array_equal(centered_context(values+500,60,32),centered_context(values,60,32))
    assert centered_context(values,60,32)[-1]==0


@pytest.mark.parametrize('origin,length',[(5,7),(100,8),(20,1)])
def test_context_boundary_rejected(origin,length):
    with pytest.raises(ValueError):centered_context(np.arange(100.),origin,length)


def test_context_warmup_is_not_filled():
    x=np.arange(100.);x[10]=np.nan
    with pytest.raises(ValueError):centered_context(x,20,15)


@pytest.mark.parametrize('n,step',[(1443,5),(732,1),(444,1)])
def test_common_schedule_has_mature_cohorts_and_unscored_endpoint(n,step):
    origins,start=schedule(n,step)
    assert len(origins)==81 and origins[-1]==n-1
    assert len(origins[(origins>=start)&(origins+5<n)])==40
    assert origins[-2]+5==n-1


def tiny_ledger():
    rows=[]
    for t in range(70):
        for h in range(1,6):
            for model in ['filter_null','same']:
                rows.append({'dataset_id':'A','group':'stocks','track':'returns','model':model,
                    'origin_pos':t,'horizon':h,'prediction':0.,'actual':float(t+h),
                    'role':'evaluation' if 40<=t<65 else 'latest' if t==69 else 'calibration'})
    return pd.DataFrame(rows)


def test_calibration_ignores_unmatured_future_labels():
    ledger=tiny_ledger();before,_,_=residual_intervals(ledger)
    poisoned=ledger.copy();poisoned.loc[poisoned.origin_pos+5>40,'actual']=1e12
    after,_,_=residual_intervals(poisoned)
    cols=['lower80','median','upper80']
    np.testing.assert_array_equal(before[before.origin_pos==40][cols],after[after.origin_pos==40][cols])
    # The exact eligible residuals at t=40 are origins 4..35, separately by horizon.
    first=before[(before.origin_pos==40)&(before.horizon==1)]
    assert first.lower80.eq(8).all() and first.upper80.eq(33).all()


def test_paired_identical_predictions_have_exact_zero_ranges():
    result=paired_ranges(tiny_ledger(),draws=100)
    np.testing.assert_array_equal(result[['normalized_MAE_difference','lower95','upper95']],0)
