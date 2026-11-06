"""Small, explicitly adapted architectures for smoothed-feature target regression.

These are educational implementations, not reproductions of published benchmarks.
Both channels are observed feature histories; five outputs are supervised residuals
or actual cumulative returns, rather than the future of either input channel.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .neural import NeuralSpec


def advanced_models():
    return [NeuralSpec('DecompositionLinear_demo','decomposition_linear'),
            NeuralSpec('LastCenteredLinear_demo','centered_linear'),
            NeuralSpec('NBEATS_demo','nbeats_demo'),
            NeuralSpec('NHITS_demo','nhits_demo'),
            NeuralSpec('PatchTransformer_demo','patch_demo')]


class LinearAdapter(nn.Module):
    def __init__(self,lookback,channels,decompose=True):
        super().__init__()
        self.decompose=decompose
        self.temporal=nn.Linear(lookback,5)
        if decompose:self.trend=nn.Linear(lookback,5)
        self.readout=nn.Linear(channels,1)

    def forward(self,x):
        z=x.transpose(1,2)
        if self.decompose:
            # Centered within a wholly observed history, never across forecast origin.
            trend=F.avg_pool1d(F.pad(z,(2,2),mode='replicate'),5,stride=1)
            z=self.temporal(z-trend)+self.trend(trend)
        else:
            last=z[:,:,-1:].detach()
            z=self.temporal(z-last)+last
        return self.readout(z.transpose(1,2)).squeeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self,size,hidden,pool=1,knots=5,lookback=20,channels=2):
        super().__init__()
        self.pool,self.lookback,self.channels=pool,lookback,channels
        pooled=((lookback+pool-1)//pool)*channels
        self.body=nn.Sequential(nn.Linear(pooled,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.backcast=nn.Linear(hidden,size)
        self.forecast=nn.Linear(hidden,knots)

    def forward(self,residual):
        x=residual.reshape(len(residual),self.lookback,self.channels).transpose(1,2)
        # Keep (time,channel) flatten order consistent with the full residual.
        x=F.max_pool1d(x,self.pool,ceil_mode=True).transpose(1,2).flatten(1)
        h=self.body(x)
        forecast=self.forecast(h)
        if forecast.shape[-1]!=5:
            forecast=F.interpolate(forecast[:,None,:],size=5,mode='linear',align_corners=False)[:,0,:]
        return self.backcast(h),forecast


class ResidualStack(nn.Module):
    def __init__(self,lookback,channels,hidden,multirate=False):
        super().__init__()
        schedule=[(4,1),(2,3),(1,5)] if multirate else [(1,5)]*3
        self.blocks=nn.ModuleList([ResidualBlock(lookback*channels,hidden,pool,knots,lookback,channels)
                                  for pool,knots in schedule])

    def forward(self,x):
        residual=x.flatten(1)
        prediction=x.new_zeros((len(x),5))
        for block in self.blocks:
            backcast,forecast=block(residual)
            residual=residual-backcast
            prediction=prediction+forecast
        return prediction


class PatchTransformer(nn.Module):
    def __init__(self,lookback,channels,hidden):
        super().__init__()
        if lookback<5 or lookback%5 or hidden%2:
            raise ValueError('Patch demo needs lookback divisible by 5 and even hidden width')
        self.patch=5
        self.embedding=nn.Linear(5*channels,hidden)
        self.position=nn.Parameter(torch.zeros(1,lookback//5,hidden))
        layer=nn.TransformerEncoderLayer(hidden,2,dim_feedforward=2*hidden,
                                         dropout=0.,batch_first=True,activation='gelu')
        self.encoder=nn.TransformerEncoder(layer,1,enable_nested_tensor=False)
        self.head=nn.Linear((lookback//5)*hidden,5)

    def forward(self,x):
        # Joint-channel observed patches: this is NOT channel-independent PatchTST.
        patches=x.reshape(len(x),x.shape[1]//self.patch,self.patch*x.shape[2])
        tokens=self.embedding(patches)+self.position
        return self.head(self.encoder(tokens).flatten(1))


def architecture(spec,lookback,channels):
    if spec.family in {'decomposition_linear','centered_linear'}:
        return LinearAdapter(lookback,channels,spec.family=='decomposition_linear')
    if spec.family in {'nbeats_demo','nhits_demo'}:
        return ResidualStack(lookback,channels,spec.hidden,spec.family=='nhits_demo')
    if spec.family=='patch_demo':return PatchTransformer(lookback,channels,spec.hidden)
    raise ValueError('Unsupported advanced demo architecture')
