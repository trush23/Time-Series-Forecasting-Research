"""Freeze TS09 settings before any benchmark scores; never overwrite a different protocol."""
from pathlib import Path
import sys,json
from dataclasses import asdict
from hashlib import sha256
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from tslab.cross_dataset import CrossSpec
from tslab.supervised import FeatureSpec
from tslab.neural import NeuralSpec
from tslab.tabular import TabularSpec
from tslab.classical import ModelSpec
catalogue=json.loads((ROOT/'inputs/cross_dataset/catalogue.json').read_text())
stock=asdict(CrossSpec())
protocol={'id':'TS09_v1','created_for_user_date':'2026-11-01','freeze_stage':'Before notebook benchmark execution',
    'dataset_ids':[x['dataset_id'] for x in catalogue],
    'input_sha256':{x['dataset_id']:x['sha256'] for x in catalogue},
    'features':asdict(FeatureSpec()),
    'schedules':{'stocks':stock,'ELNINO':asdict(CrossSpec(outer_starts=(450,540,630),outer_step=10,inner_count=4,inner_step=10,seasonal_period=12)),
                 'CO2':asdict(CrossSpec(outer_starts=(320,360,400),outer_step=5,inner_count=4,inner_step=10,seasonal_period=12))},
    'candidates':{'classical':asdict(ModelSpec('AR5_changes','ar_diff',lags=5)),
        'tabular':[asdict(TabularSpec('Ridge_10','ridge',10.)),asdict(TabularSpec('HistBoost','boost'))],
        'neural':[asdict(NeuralSpec('GRU','gru')),asdict(NeuralSpec('NBEATS_demo','nbeats_demo'))],
        'additional_monthly_candidate':'Repeat past 12 fast-EMA differences; invert EMA for actual-change forecast'},
    'selection':'Per dataset and track, equal-horizon relative MAE on past inner origins; null always eligible; min success .8',
    'evaluation':'Only outer origins with all five labels present, plus latest wholly unscored origin',
    'aggregation':'Equal-dataset average of within-dataset skill; stocks and nonfinancial separate; no pooled physical-unit errors',
    'uncertainty':'Paired moving-block MAE ranges at lengths 1/2/4; stock panel resamples same time blocks across all assets; fixed baseline normalizers',
    'independence':{'chronological_information_audit':'required','pristine_asset_or_period_holdout':False,
        'reason':'Existing project stock caches and public teaching datasets; historical exposure, selection and vintage effects remain.',
        'prospective_evaluation_completed':False},
    'prospective_plan':{'status':'specification only, not started or scheduled','first_eligible_origin_on_or_after':'2026-11-02',
        'cohort':'Same six stocks; first 60 common completed-session forecast origins after collection begins',
        'procedure':'Keep TS09 model menu, seeds, features and training budget; select by six earlier origins 20 sessions apart at first origin and every 200 common sessions; refit at every origin',
        'requirements':'Store newly issued forecast and input-vintage hashes before outcomes; score only after all five outcomes mature; no retrospective replacement or outer-score model promotion',
        'success_rule':'No paper_validated promotion from this historical audit; require a separately reviewed prospective acceptance rule before deployment'}}
data=json.dumps(protocol,indent=2).encode();path=ROOT/'TS09_PROTOCOL.json'
if path.exists() and path.read_bytes()!=data:raise ValueError('Frozen TS09 protocol differs; do not overwrite after viewing scores')
path.write_bytes(data)
print('Frozen TS09 protocol SHA256:',sha256(data).hexdigest())
