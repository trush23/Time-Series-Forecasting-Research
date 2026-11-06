"""Dataset isolation, seasonal algebra, frozen timing and synchronized inference."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from tslab.cross_dataset import CrossSpec,CrossRunner,seasonal_changes,nested_cross,macro_scores,synchronized_panel_intervals


def signal(n=350):
    return pd.Series(4+np.random.default_rng(17).normal(0,.01,n).cumsum(),index=pd.bdate_range('2020-01-01',periods=n))


def test_seasonal_ema_inversion():
    channels=np.random.default_rng(8).normal(size=(100,2))
    ds,raw=seasonal_changes(channels)
    implied=np.diff(np.r_[0.,raw]);last=channels[-1,0];out=[]
    for innovation in implied:
        last=.8*last+.2*innovation;out.append(last)
    np.testing.assert_allclose(np.cumsum(out),ds,atol=1e-14)
    np.testing.assert_allclose(ds,np.cumsum(channels[-12:-7,0]))


def test_cross_forecast_prefix_poison_and_complete_outer_cohort():
    x=signal();poison=x.copy();poison.iloc[281:]+=100
    spec=CrossSpec(outer_starts=(260,290,320),outer_step=10,inner_count=2,inner_step=10,seeds=(1,),neural_epochs=2,seasonal_period=12)
    a=CrossRunner(x,spec).predictions(280);b=CrossRunner(poison,spec).predictions(280)
    pd.testing.assert_frame_equal(a,b)
    run=nested_cross(x,spec)
    assert (run['selections'].inner_last_label<=run['selections'].selection_cutoff).all()
    assert run['scores'].n.eq(9).all()
    assert run['forecasts'].query('origin_pos==349').status.eq('immature').all()
    assert (run['tabular_audit'].max_training_label<=run['tabular_audit'].origin_pos).all()
    assert (run['neural_audit'].max_training_label<=run['neural_audit'].origin_pos).all()


def test_equal_dataset_skill_not_pooled_raw_errors():
    scores=pd.DataFrame({'dataset_id':['a','b'],'track':['returns']*2,'horizon':[5]*2,'model':['selected_pipeline']*2,
                         'MAE_skill':[.2,-.4],'MSE_skill':[.3,-.5]})
    card=macro_scores(scores,[{'dataset_id':'a','group':'stocks'},{'dataset_id':'b','group':'stocks'}]).iloc[0]
    assert card.mean_MAE_skill==pytest.approx(-.1)
    assert card.mean_MSE_skill==pytest.approx(-.1) and card.positive_datasets==1


def test_panel_bootstrap_does_not_treat_duplicate_assets_as_independent():
    rows=[];rng=np.random.default_rng(7)
    for track,base in [('smoothed','filter_nochange'),('returns','zero')]:
        e=rng.normal(size=30)
        for identifier in ['a','b']:
            for i in range(30):
                for model,error in [(base,e[i]),('selected_pipeline',e[i]*.9)]:
                    rows.append({'dataset_id':identifier,'track':track,'horizon':5,'status':'scored',
                                 'origin':pd.Timestamp('2020-01-01')+pd.Timedelta(days=i),'model':model,'error':error})
    f=pd.DataFrame(rows)
    one=synchronized_panel_intervals(f,['a'],draws=500)
    two=synchronized_panel_intervals(f,['a','b'],draws=500)
    np.testing.assert_array_equal(one[['mean_normalized_MAE_difference','lower95','upper95']],two[['mean_normalized_MAE_difference','lower95','upper95']])
    assert one.n_origins.eq(30).all() and two.n_origins.eq(30).all()


@pytest.mark.parametrize('kw',[{'outer_starts':(3,2)},{'max_samples':2},{'seeds':(1,1)},{'seasonal_period':7}])
def test_cross_spec_rejects_invalid_settings(kw):
    with pytest.raises(ValueError):CrossSpec(**kw)


@pytest.mark.parametrize('changed',['source','output','protocol'])
def test_completed_model_cache_rejects_edits(tmp_path,changed):
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_cache import save_cache,load_cache,NAMES,filename
    (tmp_path/'src/tslab').mkdir(parents=True);code=tmp_path/'src/tslab/model.py';code.write_text('VERSION=1')
    folder=tmp_path/'outputs';folder.mkdir()
    for name in NAMES:pd.DataFrame({'origin':['2020-01-01'],'value':[1]}).to_csv(folder/filename(name),index=False)
    save_cache(tmp_path,folder,'protocol_a')
    assert len(load_cache(tmp_path,folder,'protocol_a'))==len(NAMES)
    if changed=='source':code.write_text('VERSION=2')
    if changed=='output':(folder/filename('forecasts')).write_text('origin,value\n2020-01-01,9\n')
    with pytest.raises(ValueError):load_cache(tmp_path,folder,'protocol_b' if changed=='protocol' else 'protocol_a')


def test_stock_calendar_mismatch_is_not_silently_filled(tmp_path):
    import json
    from hashlib import sha256
    from tslab.cross_dataset import load_cross_inputs
    items=[]
    for name,dates in [('a',pd.date_range('2020-01-01',periods=50)),('b',pd.date_range('2020-01-01',periods=51).delete(20))]:
        frame=pd.DataFrame({'Date':dates,'value':np.arange(50.)})
        path=tmp_path/f'{name}.csv';frame.to_csv(path,index=False)
        m={'dataset_id':name,'group':'stocks','sha256':sha256(path.read_bytes()).hexdigest(),
           'rows':50,'first':str(dates[0].date()),'last':str(dates[-1].date())}
        (tmp_path/f'{name}.metadata.json').write_text(json.dumps(m));items.append(m)
    (tmp_path/'catalogue.json').write_text(json.dumps(items))
    with pytest.raises(ValueError,match='calendars differ'):load_cross_inputs(tmp_path)


def test_model_cache_accepts_only_version_only_initializer_changes(tmp_path):
    import sys,shutil
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from benchmark_cache import save_cache,load_cache,NAMES,filename
    package=tmp_path/'src/tslab';package.mkdir(parents=True)
    init=package/'__init__.py';init.write_text('__version__="1.0"\nVALUE=1\n')
    folder=tmp_path/'outputs';folder.mkdir()
    for name in NAMES:pd.DataFrame({'value':[1]}).to_csv(folder/filename(name),index=False)
    save_cache(tmp_path,folder,'p')
    historical=tmp_path/'provenance/milestone_6_sources/src/tslab/__init__.py';historical.parent.mkdir(parents=True)
    shutil.copyfile(init,historical)
    init.write_text('__version__="2.0"\nVALUE=1\n');(package/'new_unimported_module.py').write_text('EXTRA=1')
    assert load_cache(tmp_path,folder,'p') is not None
    init.write_text('__version__="2.0"\nVALUE=2\n')
    with pytest.raises(ValueError):load_cache(tmp_path,folder,'p')
