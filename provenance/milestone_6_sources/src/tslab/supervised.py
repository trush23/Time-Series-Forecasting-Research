"""Causal smoothed-input examples and five-horizon label-maturity contracts."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import validate_series
from .smoothing import FilterSpec, smooth


@dataclass(frozen=True)
class FeatureSpec:
    lookback: int = 20
    fast_alpha: float = .2
    slow_alpha: float = .05
    warmup: int = 120

    def __post_init__(self):
        if (isinstance(self.lookback,bool) or not isinstance(self.lookback,int) or self.lookback<2
            or not isinstance(self.warmup,int) or self.warmup<2):
            raise ValueError("Require integer lookback>=2 and warmup>=2")
        if not 0<self.slow_alpha<=self.fast_alpha<1:
            raise ValueError("Require 0<slow_alpha<=fast_alpha<1")


def ema_anchor(last_fast_change, alpha=.2):
    """No-change conditional mean catch-up, expressed in smoothed inputs only.

    For an EMA, x_t-s_t = (1-alpha)/alpha * (s_t-s_(t-1)). This identity
    gives the existing filter_nochange baseline without supplying raw predictors.
    """
    q=1-alpha
    return np.asarray(last_fast_change)[...,None]*(q/alpha)*(1-q**np.arange(1,6))


class SupervisedData:
    """Sequence ends at each feature origin; labels end at origin+5.

    Features are ONLY two EMA first differences. Raw prices are used to construct
    causal smoothers and actual-return labels, never as a learned predictor.
    The smoothed target uses the milestone 1 primary EMA (warmup 60).
    """
    def __init__(self, log_price, spec=FeatureSpec()):
        validate_series(log_price)
        self.log_price,self.spec=log_price,spec
        x=log_price.to_numpy()
        fast=smooth(x,FilterSpec(kind='ema',alpha=spec.fast_alpha,warmup=60))
        slow=smooth(x,FilterSpec(kind='ema',alpha=spec.slow_alpha,warmup=spec.warmup))
        self.smoothed=pd.Series(fast,index=log_price.index)
        self.channels=np.column_stack([np.r_[np.nan,np.diff(fast)],np.r_[np.nan,np.diff(slow)]])
        n=len(x)
        self.features=np.full((n,spec.lookback,2),np.nan)
        for i in range(spec.lookback-1,n):
            self.features[i]=self.channels[i-spec.lookback+1:i+1]
        self.valid_features=np.isfinite(self.features).all(axis=(1,2))
        self.anchors=ema_anchor(self.channels[:,0],spec.fast_alpha)
        self.labels={track:np.full((n,5),np.nan) for track in ['smoothed','returns']}
        for h in range(1,6):
            self.labels['smoothed'][:-h,h-1]=fast[h:]-fast[:-h]-self.anchors[:-h,h-1]
            self.labels['returns'][:-h,h-1]=x[h:]-x[:-h]

    def eligible(self, as_of, max_samples=252):
        if (isinstance(as_of,bool) or not isinstance(as_of,(int,np.integer)) or
            not 0<=as_of<len(self.log_price) or not isinstance(max_samples,int) or max_samples<40):
            raise ValueError("Invalid origin or training budget")
        origins=np.arange(len(self.log_price))
        mask=(origins+5<=as_of)&self.valid_features
        for target in self.labels.values():
            mask &= np.isfinite(target).all(axis=1)
        return origins[mask][-max_samples:]

    def training(self, as_of, track, max_samples=252):
        if track not in self.labels:
            raise ValueError("Unknown research track")
        origins=self.eligible(as_of,max_samples)
        if len(origins)<40 or not self.valid_features[as_of]:
            raise ValueError("Insufficient mature training examples")
        return self.features[origins].copy(),self.labels[track][origins].copy(),origins

    def decode(self, residual, origin, track):
        residual=np.asarray(residual,dtype=float)
        if residual.shape!=(5,) or not np.isfinite(residual).all():
            raise ValueError("Need a finite five-output prediction")
        return (self.smoothed.iloc[origin]+self.anchors[origin]+residual
                if track=='smoothed' else residual)


def chronological_holdout(origins, fraction=.2):
    """Purge optimization labels that reach the early-stop validation origin.

    Training label-end < first validation feature-origin, not merely disjoint rows.
    All validation labels must separately satisfy the outer as-of eligibility rule.
    """
    origins=np.asarray(origins)
    if len(origins)<40 or not 0<fraction<.5 or np.any(np.diff(origins)<=0):
        raise ValueError("Need >=40 increasing origins and fraction in (0,.5)")
    split=int(np.floor(len(origins)*(1-fraction)))
    validation=np.arange(split,len(origins))
    train=np.flatnonzero(origins+5<origins[split])
    if len(train)<20 or len(validation)<5:
        raise ValueError("Chronological split is too small after purging")
    return train,validation


def recursive_rollout(predict_one, sequence, track, spec=FeatureSpec()):
    """No teacher forcing: future features are generated from model predictions.

    EMA differences obey ds_(t+1)=q*ds_t+alpha*r_(t+1). For the smoothed
    residual head, residual_1=alpha_fast*r_(t+1); this identity advances BOTH
    channels without actual future observations. Return output is cumulative.
    """
    state=np.asarray(sequence,dtype=float).copy()
    if state.shape!=(spec.lookback,2) or not np.isfinite(state).all() or track not in {'smoothed','returns'}:
        raise ValueError("Invalid recursive sequence or track")
    anchor=ema_anchor(state[-1,0],spec.fast_alpha)
    fast_changes=[];returns=[]
    for _ in range(5):
        prediction=float(predict_one(state))
        if not np.isfinite(prediction):
            raise ValueError("Nonfinite one-step prediction")
        raw_return=prediction/spec.fast_alpha if track=='smoothed' else prediction
        new=np.array([(1-spec.fast_alpha)*state[-1,0]+spec.fast_alpha*raw_return,
                      (1-spec.slow_alpha)*state[-1,1]+spec.slow_alpha*raw_return])
        fast_changes.append(new[0]);returns.append(raw_return)
        state=np.vstack([state[1:],new])
    return np.cumsum(fast_changes)-anchor if track=='smoothed' else np.cumsum(returns)
