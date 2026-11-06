"""Prequential joint-residual bridge into the explicit NB9 next-close target clock."""
import numpy as np
import pandas as pd
from .integration import delayed_samples,summarize_sample


def bridge_ledger(ledger,origins,market_index,shared_support,calibration_window=126):
    rows,audits,samples=[] ,[],[]
    latest=int(ledger.origin_pos.max())
    schedule=list(map(int,origins))+([latest] if latest not in origins else [])
    for (track,model),part in ledger.groupby(['track','model']):
        point=part.pivot(index='origin_pos',columns='horizon',values='prediction').sort_index()
        error=part.pivot(index='origin_pos',columns='horizon',values='error').sort_index()
        actual=part.pivot(index='origin_pos',columns='horizon',values='actual').sort_index()
        status=part.groupby('origin_pos').fit_status.first()
        name='TS_Ridge_empirical' if model=='Ridge_10' else 'TS_null_empirical'
        for t in schedule:
            eligible=error.loc[error.index+5<=t].dropna().tail(calibration_window)
            if len(eligible)!=calibration_window:raise ValueError('Need complete mature issued-error vectors')
            native=point.loc[t].to_numpy()-eligible.to_numpy()
            delayed=delayed_samples(native)
            truth=actual.loc[t].to_numpy()[1:]-actual.loc[t].to_numpy()[0]
            support=bool(shared_support.loc[market_index[t]] and status.loc[t] in ['ok','baseline'])
            audits.append({'date':market_index[t],'origin_pos':t,'track':track,'model':name,
                'calibration_n':len(eligible),'calibration_first_origin_pos':int(eligible.index[0]),
                'calibration_last_origin_pos':int(eligible.index[-1]),'calibration_last_label_pos':int(eligible.index[-1]+5),
                'support_source':'legacy NB9 shared overlay; plus current fit availability','fit_status':status.loc[t]})
            for h in range(1,5):
                y=truth[h-1];end=t+1+h
                rows.append({'date':market_index[t],'origin_pos':t,'track':track,'horizon':h,'model':name,
                    'label_end':market_index[end] if end<len(market_index) else pd.NaT,
                    'target_start':market_index[t+1] if t+1<len(market_index) else pd.NaT,
                    'actual':y,'supported':support,'forecast_steps':h+1,'status':'scored' if np.isfinite(y) else 'immature',
                    **summarize_sample(delayed[:,h-1],actual=y if np.isfinite(y) else None)})
                if t==latest:
                    for j,value in enumerate(delayed[:,h-1]):samples.append({'date':market_index[t],'track':track,'model':name,
                        'sample_id':j,'horizon':h,'delayed_sample':value,'calibration_origin_pos':int(eligible.index[j])})
            if model!='Ridge_10':
                native_point=point.loc[t].to_numpy();point_delay=native_point[1:]-native_point[0]
                for h in range(1,5):
                    y=truth[h-1];end=t+1+h
                    rows.append({'date':market_index[t],'origin_pos':t,'track':track,'horizon':h,'model':model,
                        'label_end':market_index[end] if end<len(market_index) else pd.NaT,
                        'target_start':market_index[t+1] if t+1<len(market_index) else pd.NaT,'actual':y,
                        'supported':bool(shared_support.loc[market_index[t]]),'forecast_steps':h+1,
                        'status':'scored' if np.isfinite(y) else 'immature',**summarize_sample(np.array([point_delay[h-1]]),actual=y if np.isfinite(y) else None)})
    return pd.DataFrame(rows),pd.DataFrame(audits),pd.DataFrame(samples)


def bridge_scorecard(records):
    rows=[]
    for (track,h),part in records[records.status=='scored'].groupby(['track','horizon']):
        base='filter_nochange' if track=='smoothed' else 'zero'
        pivot=part.pivot(index='date',columns='model',values='actual')
        if pivot.isna().any().any() or not np.allclose(pivot.to_numpy(),pivot.iloc[:,0].to_numpy()[:,None]):raise ValueError('Unmatched target/origin comparison')
        baseline=part[part.model==base];base_mse=np.mean((baseline['mean']-baseline.actual)**2)
        for model,frame in part.groupby('model'):
            rows.append({'track':track,'horizon':h,'model':model,'n':len(frame),
                'MAE':frame.absolute_error.mean(),'RMSE':np.sqrt(frame.squared_error.mean()),'CRPS':frame.CRPS.mean(),
                'coverage80':frame.coverage80.mean(),'width80':frame.width80.mean(),
                'MSE_skill':1-frame.squared_error.mean()/base_mse})
    return pd.DataFrame(rows)
