"""FFT endpoint demonstrations with separate actual-return calibration."""
import numpy as np
import pandas as pd
from .data import validate_series
from .targets import make_targets
from .spectral import SpectralSpec, SpectralEndpoint, harmonic_forecast
from .classical import fit_return_bridge, bridge_forecast


def spectral_benchmark(log_price, spec=SpectralSpec(), start=500, step=10,
                       harmonic_window=128, harmonics=2):
    validate_series(log_price)
    if (not isinstance(start,int) or not isinstance(step,int) or step<1 or
        not spec.window+harmonic_window+2<=start<len(log_price)):
        raise ValueError("Start must allow both endpoint and harmonic training history")
    engine, endpoints, outputs, audit = SpectralEndpoint(spec),[],[],[]
    x=log_price.to_numpy()
    for t,value in enumerate(x):
        endpoints.append(engine.update(value)['reconstructed'])
        if t<start or ((t-start)%step!=0 and t!=len(x)-1):
            continue
        s=np.asarray(endpoints)
        nochange=np.array([row['reconstructed'] for row in engine.project(np.repeat(x[t],5))])
        drift=np.diff(x[max(0,t-60):t+1]).mean()
        driftpath=np.array([row['reconstructed'] for row in engine.project(x[t]+drift*np.arange(1,6))])
        error=""
        try:
            native,frequencies=harmonic_forecast(s[np.isfinite(s)],window=harmonic_window,harmonics=harmonics)
            bridge=fit_return_bridge(x[:t+1],s,train_window=252)
            ret=bridge_forecast(native,s,bridge)
        except Exception as exc:
            native,ret=nochange,np.zeros(5)
            frequencies={"frequencies":[]}
            error=f"{type(exc).__name__}: {exc}"
        audit.append({"origin":log_price.index[t],"origin_pos":t,"fit_status":"fallback" if error else "ok",
                      "frequencies":str(frequencies['frequencies']),"error":error,
                      "fitted_endpoint_gap":frequencies.get('fitted_endpoint_gap',np.nan),
                      "forecast_step1_change":float(native[0]-s[-1])})
        slope=s[-1]-s[-2]
        for h in range(1,6):
            predictions={'smoothed':{'persistence':s[-1],'filter_nochange':nochange[h-1],
                'filter_drift':driftpath[h-1],'smooth_slope':s[-1]+h*slope,'Fourier_plus_bridge':native[h-1]},
                'returns':{'zero':0.,'raw_drift':h*drift,'smooth_slope':h*slope,'Fourier_plus_bridge':ret[h-1]}}
            for track,methods in predictions.items():
                for model,prediction in methods.items():
                    outputs.append({"origin":log_price.index[t],"origin_pos":t,"track":track,"horizon":h,
                        "model":model,"prediction":prediction,"training_cutoff":log_price.index[t],
                        "data_available_through":log_price.index[t],
                        "fit_status":("fallback" if error else "ok") if model=='Fourier_plus_bridge' else 'baseline'})
    smoothed=pd.Series(endpoints,index=log_price.index)
    targets=make_targets(log_price,smoothed).drop(columns='origin_pos')
    records=pd.DataFrame(outputs).merge(targets,on=['origin','track','horizon'],validate='many_to_one')
    records['filter_id']=f'FFT_w{spec.window}_band_{spec.low}_{spec.high}'
    records['error']=records.prediction-records.actual
    records['units']=np.where(records.track=='returns','log_return','log_level')
    records['status']=np.where(records.actual.notna(),'scored','immature')
    return records,smoothed,pd.DataFrame(audit)
