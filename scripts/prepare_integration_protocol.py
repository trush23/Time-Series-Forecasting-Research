"""Freeze TS10 scope/rules before new model or strategy results are inspected."""
from pathlib import Path
from hashlib import sha256
import json
ROOT=Path(__file__).resolve().parents[1];FOLDER=ROOT/'inputs/nb_integration'
ref=json.loads((FOLDER/'reference_manifest.json').read_text());params=json.loads((FOLDER/'nb10/decision_configuration.json').read_text())
protocol={'id':'TS10_v1','created_for_user_date':'2026-11-01','freeze_stage':'Before TS10 new model/strategy evaluation',
    'reference_manifest_sha256':sha256((FOLDER/'reference_manifest.json').read_bytes()).hexdigest(),
    'market_sha256':ref['market_sha256'],'native_max_steps':5,'entry_delay_sessions':1,
    'library_holding_horizons':[1,2,3,4],'legacy_reference_horizons':[1,2,3,4,5],
    'primary_horizon':3,'features':{'lookback':20,'fast_alpha':.2,'slow_alpha':.05,'warmup':120},
    'point_model':'Fixed direct Ridge_10; no retrospective family selection',
    'training':{'max_samples':252,'forecast_start':650,'five_label_maturity':True},
    'calibration_window':126,'distribution':'Equal-weight joint five-error vectors from issued forecasts; subtract sample endpoints before quantiling',
    'comparison_models':['analog','recent','TS_Ridge_empirical','TS_null_empirical'],
    'common_support':'NB9 frozen metric support overlay for all four; Ridge additionally requires current fit availability',
    'decision_parameters':params,'primary_schedule':'offset 0; every three origins anchored to first frozen NB9 evaluation date',
    'sensitivity':'All holding horizons 1..4 and every offset; legacy h5 retained as six-step reference only; cost grid with decisions and allocation fractions fixed',
    'bootstrap':{'draws':2000,'seed':1731,'daily_block_lengths':[10,20,40],'comparison':'paired net daily returns versus legacy recent'},
    'claims':'Historical adjusted-price research simulation; no actual fills, orders, current-market forecast or independently validated trading advantage'}
path=ROOT/'TS10_PROTOCOL.json';data=json.dumps(protocol,indent=2).encode()
if path.exists() and path.read_bytes()!=data:raise ValueError('Frozen integration protocol differs')
path.write_bytes(data);print('Frozen TS10 protocol SHA256:',sha256(data).hexdigest())
