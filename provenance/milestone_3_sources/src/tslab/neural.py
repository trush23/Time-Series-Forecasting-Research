"""Compact deterministic-CPU neural models with purged chronological stopping.

No random train/validation split, bidirectional recurrence, teacher forcing,
future covariates, warm-started fits or best-seed selection is used.
"""
from copy import deepcopy
from dataclasses import dataclass,asdict
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .supervised import chronological_holdout


@dataclass(frozen=True)
class NeuralSpec:
    name: str
    family: str
    hidden: int = 12
    max_epochs: int = 30
    patience: int = 5
    learning_rate: float = .005
    weight_decay: float = .001
    validation_fraction: float = .2

    def __post_init__(self):
        if self.family not in {'mlp','lstm','gru','tcn'}:
            raise ValueError('Unknown neural family')
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in [self.hidden,self.max_epochs,self.patience]):
            raise ValueError('Hidden size, epochs and patience must be positive integers')
        if not 0<self.validation_fraction<.5 or self.learning_rate<=0 or self.weight_decay<0:
            raise ValueError('Invalid neural optimization settings')


def neural_models():
    return [NeuralSpec('MLP','mlp'),NeuralSpec('LSTM','lstm'),
            NeuralSpec('GRU','gru'),NeuralSpec('CausalTCN','tcn')]


def deterministic_cpu(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class CausalBlock(nn.Module):
    def __init__(self,inputs,outputs,dilation):
        super().__init__()
        self.left=2*dilation
        self.conv=nn.Conv1d(inputs,outputs,kernel_size=3,dilation=dilation)

    def forward(self,x):
        return F.relu(self.conv(F.pad(x,(self.left,0))))


class ForecastNet(nn.Module):
    def __init__(self,spec,lookback=20,channels=2):
        super().__init__()
        self.family=spec.family
        h=spec.hidden
        if self.family=='mlp':
            self.body=nn.Sequential(nn.Flatten(),nn.Linear(lookback*channels,h),nn.ReLU(),
                                    nn.Linear(h,h),nn.ReLU())
        elif self.family in {'lstm','gru'}:
            recurrent=nn.LSTM if self.family=='lstm' else nn.GRU
            self.body=recurrent(channels,h,batch_first=True,bidirectional=False)
        else:
            self.body=nn.Sequential(CausalBlock(channels,h,1),CausalBlock(h,h,2),
                                    CausalBlock(h,h,4),CausalBlock(h,h,8))
        self.head=nn.Linear(h,5)

    def forward(self,x):
        if self.family=='mlp':
            hidden=self.body(x)
        elif self.family in {'lstm','gru'}:
            hidden=self.body(x)[0][:,-1,:]
        else:
            hidden=self.body(x.transpose(1,2))[:,:,-1]
        return self.head(hidden)


def _scales(X,y):
    # Pool time positions within each channel, using optimization-training rows only.
    xm=X.mean(axis=(0,1),keepdims=True);xs=X.std(axis=(0,1),keepdims=True)
    ym=y.mean(axis=0,keepdims=True);ys=y.std(axis=0,keepdims=True)
    return xm,np.where(xs>1e-12,xs,1.),ym,np.where(ys>1e-12,ys,1.)


class NeuralFit:
    def __init__(self,model,scales,spec,lookback,seed,epochs):
        self.model,self.scales,self.spec=model,scales,spec
        self.lookback,self.seed,self.epochs=lookback,seed,epochs

    def predict(self,sequence):
        xm,xs,ym,ys=self.scales
        X=np.asarray(sequence,dtype=float)[None,:,:]
        self.model.eval()
        with torch.no_grad():
            y=self.model(torch.tensor((X-xm)/xs,dtype=torch.float32)).numpy()
        return (y*ys+ym)[0]

    def checkpoint(self):
        return {'schema_version':1,'state_dict':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
                'scales':[torch.tensor(v,dtype=torch.float64) for v in self.scales],
                'spec':asdict(self.spec),'lookback':self.lookback,'channels':self.scales[0].shape[-1],
                'seed':self.seed,'epochs':self.epochs}

    @classmethod
    def restore(cls,state):
        if state.get('schema_version')!=1:
            raise ValueError('Unsupported neural checkpoint')
        spec=NeuralSpec(**state['spec'])
        net=ForecastNet(spec,state['lookback'],state['channels'])
        net.load_state_dict(state['state_dict'])
        return cls(net,tuple(v.numpy() for v in state['scales']),spec,state['lookback'],state['seed'],state['epochs'])


def fit_neural(X,y,origins,spec,seed=1731):
    """Choose epochs on a purged past tail, reset, then refit all mature rows.

    Stopping-stage scalers use optimization training rows only. After epoch choice,
    final scalers use all mature examples and a fresh model is trained for exactly
    the chosen number of epochs. All fits are full-batch, AdamW, gradient norm<=1.
    """
    X,y=np.asarray(X,dtype=float),np.asarray(y,dtype=float)
    if X.ndim!=3 or X.shape[2]!=2 or y.shape!=(len(X),5) or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Invalid finite sequence/target arrays')
    origins=np.asarray(origins)
    train,val=chronological_holdout(origins,spec.validation_fraction)
    stop_scales=_scales(X[train],y[train])
    xm,xs,ym,ys=stop_scales
    train_x=torch.tensor((X[train]-xm)/xs,dtype=torch.float32)
    train_y=torch.tensor((y[train]-ym)/ys,dtype=torch.float32)
    val_x=torch.tensor((X[val]-xm)/xs,dtype=torch.float32)
    val_y=torch.tensor((y[val]-ym)/ys,dtype=torch.float32)
    deterministic_cpu(seed)
    model=ForecastNet(spec,X.shape[1],X.shape[2])
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec.learning_rate,weight_decay=spec.weight_decay)
    best,best_epoch,wait=np.inf,0,0
    trace=[]
    for epoch in range(1,spec.max_epochs+1):
        model.train();optimizer.zero_grad()
        loss=F.mse_loss(model(train_x),train_y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural training loss')
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval()
        with torch.no_grad():validation=float(F.mse_loss(model(val_x),val_y))
        if not np.isfinite(validation):raise RuntimeError('Nonfinite chronological validation loss')
        trace.append({'epoch':epoch,'training_MSE_standardized':float(loss.detach()),
                      'validation_MSE_standardized':validation})
        if validation<best-1e-8:
            best,best_epoch,wait=validation,epoch,0
        else:wait+=1
        if wait>=spec.patience:break
    # No held-out future observation enters this refit: all examples mature by origin.
    final_scales=_scales(X,y)
    xm,xs,ym,ys=final_scales
    full_x=torch.tensor((X-xm)/xs,dtype=torch.float32)
    full_y=torch.tensor((y-ym)/ys,dtype=torch.float32)
    deterministic_cpu(seed)
    final_model=ForecastNet(spec,X.shape[1],X.shape[2])
    optimizer=torch.optim.AdamW(final_model.parameters(),lr=spec.learning_rate,weight_decay=spec.weight_decay)
    for _ in range(best_epoch):
        optimizer.zero_grad();loss=F.mse_loss(final_model(full_x),full_y)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural refit loss')
        loss.backward();nn.utils.clip_grad_norm_(final_model.parameters(),1.);optimizer.step()
    fitted=NeuralFit(final_model,final_scales,spec,X.shape[1],seed,best_epoch)
    return fitted,{'n_train':len(X),'optimization_rows':len(train),'validation_rows':len(val),
        'optimization_last_label':int(origins[train[-1]]+5),
        'validation_first_origin':int(origins[val[0]]),'last_training_label':int(origins[-1]+5),
        'chosen_epochs':best_epoch,'epochs_examined':len(trace),
        'parameter_count':sum(p.numel() for p in final_model.parameters()),
        'stopping_feature_mean':stop_scales[0].reshape(-1).tolist(),
        'feature_mean':final_scales[0].reshape(-1).tolist(),
        'trace':trace}
