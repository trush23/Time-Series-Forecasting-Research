"""Export learned experiments and audit matched-date frozen comparisons."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import digest
from .evaluation import scorecard,paired_block_difference


def export_learning(run,folder):
    folder=Path(folder)
    names=['forecasts','inner_forecasts','selections','tuning','fit_audit','learning_curves','seed_forecasts']
    for name in names:
        if run[name].empty:continue
        filename='historical_forecasts.csv' if name=='forecasts' else name+'.csv'
        run[name].to_csv(folder/filename,index=False)
    run['smoothed'].to_csv(folder/'causal_smoothed_target.csv',index_label='Date')
    run['feature_channels'].to_csv(folder/'causal_feature_channels.csv',index_label='Date')
    scores=scorecard(run['forecasts'])
    for track in ['smoothed','returns']:
        scores[scores.track==track].to_csv(folder/f'{track}_scorecard.csv',index=False)
    intervals=[]
    for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
        for block in [1,2,4]:
            intervals.append({'track':track,**paired_block_difference(run['forecasts'],track,
                'selected_pipeline',baseline,horizon=5,block=block,draws=2000)})
    intervals=pd.DataFrame(intervals)
    intervals.to_csv(folder/'selected_paired_uncertainty.csv',index=False)
    if not run['seed_forecasts'].empty:
        members=run['seed_forecasts'].copy()
        target=run['forecasts'][['origin','track','horizon','actual','status','filter_id']].drop_duplicates()
        members=members.merge(target,on=['origin','track','horizon'],how='inner',validate='many_to_one')
        members['family']=members.model
        members['model']=members.model+'_seed'+members.seed.astype(str)
        members['error']=members.prediction-members.actual
        base=run['forecasts'][run['forecasts'].model.isin(['filter_nochange','zero'])]
        seed_scores=scorecard(pd.concat([members,base],ignore_index=True))
        seed_scores=seed_scores[~seed_scores.model.isin(['filter_nochange','zero'])]
        seed_scores[['family','seed']]=seed_scores.model.str.rsplit('_seed',n=1,expand=True)
        seed_scores.to_csv(folder/'seed_scorecards.csv',index=False)
    return scores,intervals


def frozen_comparison(records,reference_paths,metadata_path,current_label):
    """Same target and exact common outer origins; no retrospective reselection.

    Prior models retain their ORIGINAL frozen block choices. Training/selection
    budgets differ between milestones, so this is not an architecture-only test.
    """
    metadata=json.loads(Path(metadata_path).read_text())
    current=records[records.model=='selected_pipeline'].copy()
    current['model']=current_label
    frames=[current]
    for label,path in reference_paths.items():
        if digest(path)!=metadata[label]['sha256']:raise ValueError('Frozen reference checksum mismatch')
        reference=pd.read_csv(path,parse_dates=['origin'])
        reference=reference[reference.model=='selected_pipeline'].copy()
        reference=reference[reference.origin.isin(current.origin.unique())]
        left=current[['origin','track','horizon','actual']]
        joined=reference.merge(left,on=['origin','track','horizon'],suffixes=('_old','_new'),validate='one_to_one')
        if len(joined)!=len(current):raise ValueError('Reference has incomplete common origins')
        np.testing.assert_allclose(joined.actual_old,joined.actual_new,equal_nan=True,atol=1e-12)
        reference['model']=label;frames.append(reference)
    base=records[records.model.isin(['filter_nochange','zero'])]
    return scorecard(pd.concat([*frames,base],ignore_index=True))
