"""Pinned foundation-model laboratory. Optional neural imports are lazy.

Chronos forecasts centered causal EMA LEVELS. Inverting its marginal median path
is a point functional, NOT a return median or a joint predictive distribution.
Both target tracks use a common mature-residual interval procedure.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from .data import digest
from .supervised import SupervisedData
from .tabular import TabularSpec, fit_tabular
from .probabilistic import interval_score, pinball


def invert_ema_path(path, current, previous, alpha=.2):
    """Map a deterministic future EMA path to actual changes using EMA algebra."""
    p = np.asarray(path, dtype=float)
    if p.shape[-1:] != (5,) or not np.isfinite(p).all() or not 0 < alpha <= 1:
        raise ValueError('Need finite five-step paths and alpha in (0,1]')
    lag = np.concatenate([np.broadcast_to(current, p.shape[:-1])[..., None], p[..., :-1]], axis=-1)
    now = (current-(1-alpha)*previous)/alpha
    return (p-(1-alpha)*lag)/alpha-np.asarray(now)[..., None]


def centered_context(smoothed, origin, length):
    a = np.asarray(smoothed, dtype=float)
    if not isinstance(length, int) or length < 2 or origin < length-1 or origin >= len(a):
        raise ValueError('Invalid context boundary')
    context = a[origin-length+1:origin+1].copy()
    if not np.isfinite(context).all():
        raise ValueError('Nonfinite context; do not fill warmup or future values')
    return context-context[-1]


def schedule(n, step, calibration_count=40, evaluation_count=40):
    last = n-6
    origins = last-np.arange(calibration_count+evaluation_count-1, -1, -1)*step
    if origins[0] < 315:
        raise ValueError('Insufficient history for common context/training budget')
    return np.r_[origins, n-1].astype(int), int(origins[calibration_count])


def verify_checkpoint_files(root, records):
    for record in records:
        for name, expected in record['files_sha256'].items():
            if digest(Path(root)/'inputs/foundation_models'/record['folder']/name) != expected:
                raise ValueError('Checkpoint file mismatch: '+name)


def foundation_quantiles(root, signals, catalogue, protocol, recompute=False):
    """Replay checked inference by default; opt in to fresh CPU inference offline."""
    root = Path(root)
    if protocol != json.loads((root/'TS11_PROTOCOL.json').read_text()):
        raise ValueError('Effective protocol differs from frozen file')
    folder = root/'outputs/TS11_Foundation_Model_Laboratory'
    folder.mkdir(parents=True, exist_ok=True)
    cache, meta = folder/'foundation_quantiles.npz', folder/'inference_manifest.json'
    checkpoints = json.loads((root/'inputs/foundation_models/checkpoint_manifest.json').read_text())
    verify_checkpoint_files(root, checkpoints)
    identity = {'protocol_sha256': digest(root/'TS11_PROTOCOL.json'),
        'module_sha256': digest(__file__),
        'checkpoint_manifest_sha256': digest(root/'inputs/foundation_models/checkpoint_manifest.json'),
        'signal_sha256': {name:__import__('hashlib').sha256(pd.util.hash_pandas_object(signal,index=True).to_numpy().tobytes()).hexdigest() for name,signal in signals.items()},
        'inputs_sha256': {name: digest(root/f'inputs/cross_dataset/{name}.csv') for name in signals}}
    if cache.exists() and not recompute:
        saved = json.loads(meta.read_text())
        if saved['identity'] != identity or saved['cache_sha256'] != digest(cache):
            raise ValueError('Inference cache provenance changed; use RECOMPUTE=True after review')
        with np.load(cache, allow_pickle=False) as z:
            arrays = {name: z[name].copy() for name in z.files}
        return arrays, saved
    if not recompute:
        raise FileNotFoundError('No audited inference cache. Install optional foundation extra; use RECOMPUTE=True')
    import torch
    from chronos import ChronosBoltPipeline
    from importlib.metadata import version
    torch.set_num_threads(1)
    torch.manual_seed(protocol['seed'])
    torch.use_deterministic_algorithms(True)
    arrays, timings = {}, []
    for checkpoint in checkpoints:
        start = time.perf_counter()
        pipeline = ChronosBoltPipeline.from_pretrained(
            str(root/'inputs/foundation_models'/checkpoint['folder']),
            device_map='cpu', torch_dtype=torch.float32, local_files_only=True)
        load_seconds = time.perf_counter()-start
        parameter_count = sum(p.numel() for p in pipeline.model.parameters())
        levels = np.asarray(pipeline.model.config.chronos_config['quantiles'])
        np.testing.assert_allclose(levels, np.arange(1,10)/10)
        for length in protocol['contexts']:
            # Exclude first-call setup from the reported repeated inference time.
            pipeline.predict(torch.zeros(1,length), prediction_length=5)
            for listed in catalogue:
                name = listed['dataset_id']
                data = SupervisedData(signals[name])
                step = protocol['stock_step'] if listed['group']=='stocks' else protocol['monthly_step']
                origins, _ = schedule(len(signals[name]), step)
                contexts = np.stack([centered_context(data.smoothed, t, length) for t in origins])
                predictions = []
                start = time.perf_counter()
                for first in range(0,len(contexts),protocol['batch_size']):
                    batch = torch.tensor(contexts[first:first+protocol['batch_size']],dtype=torch.float32)
                    with torch.inference_mode():
                        predictions.append(pipeline.predict(batch,prediction_length=5).cpu().numpy())
                elapsed = time.perf_counter()-start
                key = f"{checkpoint['folder']}_L{length}__{name}"
                arrays[key] = np.concatenate(predictions).astype(float)
                timings.append({'model': f"{checkpoint['folder']}_L{length}", 'dataset_id':name,
                    'context':length, 'origins':len(origins), 'seconds':elapsed,
                    'milliseconds_per_origin_both_tracks':1000*elapsed/len(origins),
                    'load_seconds':load_seconds, 'parameters':parameter_count,
                    'weights_bytes':(root/'inputs/foundation_models'/checkpoint['folder']/'model.safetensors').stat().st_size})
                print(f'{key}: {len(origins)} origins, {elapsed:.2f}s',flush=True)
        # Independent histories generated after checkpoint retrieval; no financial claim.
        rng = np.random.default_rng(protocol['seed']+11)
        x = np.cumsum(rng.normal(0,.01,(128,325)),axis=1)
        s = x.copy()
        for j in range(1,x.shape[1]): s[:,j]=.2*x[:,j]+.8*s[:,j-1]
        context = s[:,64:320]-s[:,319,None]
        with torch.inference_mode():
            pred = pipeline.predict(torch.tensor(context,dtype=torch.float32),prediction_length=5).cpu().numpy()
        arrays[checkpoint['folder']+'__martingale'] = pred.astype(float)
        del pipeline
    np.savez_compressed(cache, **arrays)
    saved = {'identity':identity, 'cache_sha256':digest(cache), 'timings':timings,
        'versions':{p:version(p) for p in ['torch','chronos-forecasting','transformers','accelerate','huggingface-hub']},
        'device':'CPU float32; one torch thread; batch 32; five forecast steps',
        'excluded_costs':'pretraining, dependency installation, downloads, quantile decoding and scoring'}
    meta.write_text(json.dumps(saved,indent=2))
    return arrays, saved


def replay(root, signals, catalogue, protocol, arrays):
    rows, fits, context_audit, timing = [], [], [], []
    checkpoints = json.loads((Path(root)/'inputs/foundation_models/checkpoint_manifest.json').read_text())
    for listed in catalogue:
        name = listed['dataset_id']; signal = signals[name]; data = SupervisedData(signal)
        x, s = signal.to_numpy(), data.smoothed.to_numpy()
        step = protocol['stock_step'] if listed['group']=='stocks' else protocol['monthly_step']
        origins, evaluation_start = schedule(len(signal),step)
        for k,t in enumerate(origins):
            forecasts = {}
            started = time.perf_counter()
            for track in ['smoothed','returns']:
                forecasts[(track,'filter_null')] = data.decode(np.zeros(5),t,track)
                X,y,train = data.training(t,track,protocol['ridge_samples'])
                fit, _ = fit_tabular(X,y,TabularSpec('Ridge_10','ridge',10.))
                forecasts[(track,'Ridge_10')] = data.decode(fit.predict(data.features[t]),t,track)
                fits.append({'dataset_id':name,'track':track,'origin_pos':int(t),
                    'max_training_label_pos':int(train[-1]+5),'n_train':len(train)})
            timing.append({'dataset_id':name,'origin_pos':int(t),'model':'Ridge_10',
                'milliseconds_per_origin_both_tracks':1000*(time.perf_counter()-started)})
            for checkpoint in checkpoints:
                for length in protocol['contexts']:
                    model = f"{checkpoint['folder']}_L{length}"
                    q = arrays[f'{model}__{name}'][k]
                    if q.shape != (9,5) or not np.isfinite(q).all(): raise ValueError('Invalid model output')
                    path = q[4]+s[t]
                    forecasts[('smoothed',model)] = path
                    forecasts[('returns',model)] = invert_ema_path(path,s[t],s[t-1])
                    context = centered_context(s,t,length)
                    context_audit.append({'dataset_id':name,'model':model,'origin_pos':int(t),
                        'context_start_pos':int(t-length+1),'context_end_pos':int(t),
                        'context_sha256':__import__('hashlib').sha256(context.tobytes()).hexdigest(),
                        'quantile_crossings':int((np.diff(q,axis=0)<0).sum())})
            for (track,model),prediction in forecasts.items():
                for h in range(1,6):
                    end = t+h
                    actual = (s[end] if track=='smoothed' else x[end]-x[t]) if end<len(x) else np.nan
                    rows.append({'dataset_id':name,'group':listed['group'],'track':track,'model':model,
                        'origin_pos':int(t),'origin':signal.index[t],'horizon':h,
                        'target_end':signal.index[end] if end<len(x) else pd.NaT,
                        'data_available_through':signal.index[t], 'actual':actual,
                        'prediction':prediction[h-1], 'status':'scored' if end<len(x) else 'immature',
                        'role':'evaluation' if t>=evaluation_start and end<len(x) else 'latest' if t==len(x)-1 else 'calibration',
                        'units':listed['units'] if track=='smoothed' else 'log_return' if listed['group']=='stocks' else listed['units']+' change'})
    ledger = pd.DataFrame(rows)
    return ledger,pd.DataFrame(fits),pd.DataFrame(context_audit),pd.DataFrame(timing)


def score_points(ledger):
    f = ledger[ledger.role=='evaluation'].copy()
    f['absolute_error'] = (f.prediction-f.actual).abs()
    f['squared_error'] = (f.prediction-f.actual)**2
    scores = f.groupby(['dataset_id','group','track','model','horizon']).agg(
        n=('actual','size'),MAE=('absolute_error','mean'),MSE=('squared_error','mean')).reset_index()
    reference = scores[scores.model=='filter_null'].drop(columns=['model','n']).rename(columns={'MAE':'null_MAE','MSE':'null_MSE'})
    scores = scores.merge(reference,on=['dataset_id','group','track','horizon'],validate='many_to_one')
    scores['MAE_skill']=1-scores.MAE/scores.null_MAE
    scores['MSE_skill']=1-scores.MSE/scores.null_MSE
    macro = scores.groupby(['group','track','model','horizon']).agg(
        datasets=('dataset_id','nunique'),MAE_skill=('MAE_skill','mean'),MSE_skill=('MSE_skill','mean'),
        positive_MAE_datasets=('MAE_skill',lambda a:int((a>0).sum()))).reset_index()
    return scores,macro


def residual_intervals(ledger, window=32):
    rows,audit=[],[]
    for (name,track,model),part in ledger.groupby(['dataset_id','track','model']):
        part=part.sort_values(['origin_pos','horizon'])
        pivot=part.assign(residual=part.actual-part.prediction).pivot(index='origin_pos',columns='horizon',values='residual')
        for t,now in part[part.role.isin(['evaluation','latest'])].groupby('origin_pos'):
            eligible=pivot.loc[pivot.index+5<=t].dropna().tail(window)
            if len(eligible)!=window:raise ValueError('Incomplete common mature calibration window')
            offsets=np.quantile(eligible.to_numpy(),[.1,.5,.9],axis=0,method='inverted_cdf')
            audit.append({'dataset_id':name,'track':track,'model':model,'origin_pos':t,
                'calibration_n':len(eligible),'first_calibration_origin':int(eligible.index[0]),
                'last_calibration_label':int(eligible.index[-1]+5)})
            for j,row in enumerate(now.itertuples()):
                lo,med,hi=row.prediction+offsets[:,j]
                y=row.actual
                rows.append({'dataset_id':name,'group':row.group,'track':track,'model':model,'origin_pos':t,
                    'horizon':row.horizon,'role':row.role,'actual':y,'lower80':lo,'median':med,'upper80':hi,
                    'covered80':float(lo<=y<=hi) if np.isfinite(y) else np.nan,'width80':hi-lo,
                    'interval_score80':float(interval_score(y,lo,hi,.8)) if np.isfinite(y) else np.nan,
                    'pinball10':float(pinball(y,lo,.1)) if np.isfinite(y) else np.nan,
                    'pinball90':float(pinball(y,hi,.9)) if np.isfinite(y) else np.nan})
    table=pd.DataFrame(rows)
    scores=table[table.role=='evaluation'].groupby(['dataset_id','group','track','model','horizon']).agg(
        n=('actual','size'),coverage80=('covered80','mean'),width80=('width80','mean'),
        interval_score80=('interval_score80','mean'),pinball10=('pinball10','mean'),pinball90=('pinball90','mean')).reset_index()
    return table,pd.DataFrame(audit),scores


def paired_ranges(ledger, draws=2000, seed=1731):
    """Equal-asset normalized MAE differences; synchronized time blocks per group.

    Fixed denominators and methods, h=5. Exploratory conditional ranges, no
    multiplicity adjustment, no pristine pretraining or prospective guarantee.
    """
    rng=np.random.default_rng(seed);rows=[]
    f=ledger[(ledger.role=='evaluation')&(ledger.horizon==5)].copy()
    f['loss']=(f.prediction-f.actual).abs()
    for (group,track),part in f.groupby(['group','track']):
        for model in sorted(set(part.model)-{'filter_null'}):
            columns=[]
            for name,asset in part.groupby('dataset_id'):
                a=asset.pivot(index='origin_pos',columns='model',values='loss').sort_index()
                columns.append((a[model]-a['filter_null']).to_numpy()/a['filter_null'].mean())
            delta=np.stack(columns,axis=1).mean(axis=1)
            for length in [1,4,8]:
                n=len(delta);blocks=int(np.ceil(n/length))
                starts=rng.integers(0,n-length+1,size=(draws,blocks))
                indices=(starts[...,None]+np.arange(length)).reshape(draws,-1)[:,:n]
                lo,hi=np.quantile(delta[indices].mean(axis=1),[.025,.975])
                rows.append({'group':group,'track':track,'model':model,'horizon':5,'block_origins':length,
                    'normalized_MAE_difference':delta.mean(),'lower95':lo,'upper95':hi,
                    'assets':len(columns),'origins':len(delta),'interpretation':'negative favors model; fixed baseline denominators'})
    return pd.DataFrame(rows)


def martingale_control(arrays,seed=1731):
    rng=np.random.default_rng(seed+11);sigma=.01;a=.2;q=1-a
    x=np.cumsum(rng.normal(0,sigma,(128,325)),axis=1);s=x.copy()
    for j in range(1,x.shape[1]):s[:,j]=a*x[:,j]+q*s[:,j-1]
    hs=np.arange(1,6)
    null=s[:,319,None]+(x[:,319]-s[:,319])[:,None]*(1-q**hs)
    variance=sigma**2*np.cumsum((1-q**hs)**2)
    rows=[]
    for key,quantiles in arrays.items():
        if not key.endswith('__martingale'):continue
        path=quantiles[:,4,:]+s[:,319,None]
        returns=invert_ema_path(path,s[:,319],s[:,318])
        for h in range(1,6):
            rows.append({'model':key.split('__')[0],'horizon':h,
                'smoothed_conditional_MSE_ratio':1+np.mean((path[:,h-1]-null[:,h-1])**2)/variance[h-1],
                'return_conditional_MSE_ratio':1+np.mean(returns[:,h-1]**2)/(sigma**2*h),
                'histories':128,'null_ratio':1.,'method':'exact IID Gaussian conditional risk; includes future innovation variance'})
    return pd.DataFrame(rows)
