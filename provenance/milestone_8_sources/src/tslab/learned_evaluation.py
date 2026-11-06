"""Common five-horizon supervised evaluation for tabular and neural candidates."""
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
from .supervised import SupervisedData,FeatureSpec
from .smoothing import CausalSmoother,FilterSpec
from .targets import make_targets
from .evaluation import null_predictions
from .prequential import choose_models
from .tabular import fit_tabular


class LearnedRunner:
    def __init__(self,log_price,models,mode='tabular',feature_spec=FeatureSpec(),
                 max_samples=252,seeds=(1731,1732,1733),checkpoint_dir=None):
        if mode not in {'tabular','neural'} or not models or len({m.name for m in models})!=len(models):
            raise ValueError('Need a supported mode and uniquely named models')
        if not seeds or len(set(seeds))!=len(seeds):
            raise ValueError('Neural seed list must be fixed and unique')
        self.data=SupervisedData(log_price,feature_spec)
        self.log_price,self.models,self.mode=log_price,list(models),mode
        self.max_samples,self.seeds=max_samples,tuple(seeds)
        self.checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None
        self.cache={}
        # Audit, trace and member records are append-only lists; fits cache by origin.
        self.audit,self.traces,self.members=[],[],[]
        self.feature_spec=feature_spec
        self.targets=make_targets(log_price,self.data.smoothed)

    def predictions(self,t):
        if t in self.cache:return self.cache[t].copy()
        x=self.log_price.iloc[:t+1].to_numpy()
        s=self.data.smoothed.iloc[:t+1].to_numpy()
        engine=CausalSmoother(FilterSpec(kind='ema',alpha=self.feature_spec.fast_alpha,warmup=60))
        engine.update_many(x)
        base=null_predictions(x,s,engine)
        base['fit_status']='baseline';base['chosen_model']=base.model
        outputs=[base]
        query=self.data.features[t]
        lookup=base.set_index(['track','model','horizon']).prediction
        for track in ['smoothed','returns']:
            X,y,origins=self.data.training(t,track,self.max_samples)
            for spec in self.models:
                start=time.perf_counter();diagnostic={};error='';member_predictions=[]
                try:
                    if self.mode=='tabular':
                        fitted,diagnostic=fit_tabular(X,y,spec)
                        residual=fitted.predict(query,track,self.feature_spec)
                    else:
                        from .neural import fit_neural
                        seed_errors=[]
                        for seed in self.seeds:
                            seed_error=''
                            try:
                                fit,detail=fit_neural(X,y,origins,spec,seed)
                                residual_member=fit.predict(query)
                                if not np.isfinite(residual_member).all():raise RuntimeError('Nonfinite seed prediction')
                            except Exception as exc:
                                fit,detail=None,{'trace':[],'n_train':len(origins)}
                                residual_member=np.zeros(5)
                                seed_error=f'seed {seed}: {type(exc).__name__}: {exc}'
                                seed_errors.append(seed_error)
                            member_predictions.append(residual_member)
                            for row in detail.pop('trace'):
                                self.traces.append({'origin':self.log_price.index[t],'origin_pos':t,
                                    'track':track,'model':spec.name,'seed':seed,**row})
                            self.audit.append({'origin':self.log_price.index[t],'origin_pos':t,
                                    'track':track,'model':spec.name,'seed':seed,
                                    'fit_status':'fallback' if seed_error else 'ok','error':seed_error,
                                'max_training_label':int(origins[-1]+5),**detail})
                            decoded=self.data.decode(residual_member,t,track)
                            for h in range(1,6):
                                self.members.append({'origin':self.log_price.index[t],'track':track,
                                    'horizon':h,'model':spec.name,'seed':seed,'prediction':decoded[h-1],
                                    'fit_status':'fallback' if seed_error else 'ok'})
                            if fit is not None and self.checkpoint_dir and t==len(self.log_price)-1:
                                import torch
                                self.checkpoint_dir.mkdir(parents=True,exist_ok=True)
                                checkpoint=fit.checkpoint()
                                checkpoint.update({'feature_spec':asdict(self.feature_spec),'track':track,
                                    'origin':str(self.log_price.index[t]),'origin_pos':int(t),
                                    'smoothed_origin':float(s[-1]),
                                    'anchor':torch.tensor(self.data.anchors[t],dtype=torch.float64),
                                    'sequence':torch.tensor(query,dtype=torch.float64),
                                    'residual_prediction':torch.tensor(residual_member,dtype=torch.float64),
                                    'decoded_prediction':torch.tensor(decoded,dtype=torch.float64)})
                                torch.save(checkpoint,self.checkpoint_dir/f'{spec.name}_{track}_seed{seed}.pt')
                        # All seeds must fit. A partial ensemble is not substituted silently.
                        if seed_errors:raise RuntimeError('Incomplete seed ensemble: '+' | '.join(seed_errors))
                        residual=np.mean(member_predictions,axis=0)
                        diagnostic={'n_train':len(origins),'successful_seed_count':len(member_predictions)}
                    point=self.data.decode(residual,t,track)
                    if not np.isfinite(point).all():raise RuntimeError('Nonfinite learned forecast')
                    status='ok'
                except Exception as exc:
                    status='fallback';error=f'{type(exc).__name__}: {exc}'
                    baseline='filter_nochange' if track=='smoothed' else 'zero'
                    point=np.array([lookup.loc[(track,baseline,h)] for h in range(1,6)])
                self.audit.append({'origin':self.log_price.index[t],'origin_pos':t,'track':track,
                    'model':spec.name,'seed':'ensemble' if self.mode=='neural' else 'fixed',
                    'fit_status':status,'error':error,'max_training_label':int(origins[-1]+5),
                    'last_training_feature_origin':int(origins[-1]),'first_training_feature_origin':int(origins[0]),
                    'runtime_seconds':time.perf_counter()-start,**diagnostic})
                for h in range(1,6):
                    outputs.append(pd.DataFrame([{'track':track,'horizon':h,'model':spec.name,
                        'prediction':point[h-1],'fit_status':status,'chosen_model':spec.name}]))
        result=pd.concat(outputs,ignore_index=True)
        result['origin']=self.log_price.index[t];result['origin_pos']=t
        result['training_cutoff']=self.log_price.index[t]
        result['data_available_through']=self.log_price.index[t]
        self.cache[t]=result
        return result.copy()

    def attach_labels(self,predictions):
        result=predictions.merge(self.targets.drop(columns='origin_pos'),
            on=['origin','track','horizon'],validate='many_to_one')
        result['filter_id']=f'ema_alpha_{self.feature_spec.fast_alpha}'
        result['units']=np.where(result.track=='smoothed','log_level','log_return')
        result['error']=result.prediction-result.actual
        result['status']=np.where(result.actual.notna(),'scored','immature')
        return result


def nested_learned(log_price,models,mode='tabular',feature_spec=FeatureSpec(),
                   outer_starts=(1000,1400,1800),outer_step=40,inner_count=3,inner_step=40,
                   max_samples=252,min_success=.8,seeds=(1731,1732,1733),checkpoint_dir=None,
                   progress=False):
    starts=tuple(outer_starts)
    if not starts or starts!=tuple(sorted(set(starts))) or starts[-1]>=len(log_price):
        raise ValueError('Invalid outer block boundaries')
    if min(outer_step,inner_count,inner_step)<1:raise ValueError('Invalid schedule spacing')
    runner=LearnedRunner(log_price,models,mode,feature_spec,max_samples,seeds,checkpoint_dir)
    selection,tuning,inner_exports,outer_exports=[],[],[],[]
    for block,start in enumerate(starts):
        inner=[start-6-inner_step*k for k in reversed(range(inner_count))]
        if inner[0]<feature_spec.warmup+feature_spec.lookback+45:
            raise ValueError('Insufficient history before inner validation')
        inner_records=runner.attach_labels(pd.concat([runner.predictions(i) for i in inner],ignore_index=True))
        cutoff=log_price.index[start-1]
        assert (inner_records.available_at<=cutoff).all()
        choices,table=choose_models(inner_records,[m.name for m in models],min_success)
        table['block']=block;tuning.append(table)
        inner_records['block']=block;inner_exports.append(inner_records)
        stop=starts[block+1] if block+1<len(starts) else len(log_price)
        origins=list(range(start,stop,outer_step))
        if block==len(starts)-1 and origins[-1]!=len(log_price)-1:origins.append(len(log_price)-1)
        for track,chosen in choices.items():
            selection.append({'block':block,'track':track,'chosen_model':chosen,'selection_cutoff':cutoff,
                'outer_first_origin':log_price.index[start],'outer_last_origin':log_price.index[origins[-1]],
                'inner_first_origin':log_price.index[inner[0]],'inner_last_label':log_price.index[inner[-1]+5],
                'objective':'equal-horizon relative MAE','minimum_success_fraction':min_success})
        for t in origins:
            predictions=runner.predictions(t)
            selected=pd.concat([predictions[(predictions.track==track)&(predictions.model==chosen)]
                                for track,chosen in choices.items()],ignore_index=True).copy()
            selected['chosen_model']=selected.model;selected['model']='selected_pipeline'
            predictions=pd.concat([predictions,selected],ignore_index=True)
            predictions['block']=block;predictions['selection_cutoff']=cutoff
            outer_exports.append(runner.attach_labels(predictions))
        if progress:print(f'Completed {mode} block {block+1}/{len(starts)}; choices: {choices}',flush=True)
    return {'forecasts':pd.concat(outer_exports,ignore_index=True),
        'inner_forecasts':pd.concat(inner_exports,ignore_index=True),'selections':pd.DataFrame(selection),
        'tuning':pd.concat(tuning,ignore_index=True),'fit_audit':pd.DataFrame(runner.audit),
        'learning_curves':pd.DataFrame(runner.traces),'seed_forecasts':pd.DataFrame(runner.members),
        'smoothed':runner.data.smoothed,'feature_channels':pd.DataFrame(runner.data.channels,
            index=log_price.index,columns=['fast_EMA_change','slow_EMA_change'])}
