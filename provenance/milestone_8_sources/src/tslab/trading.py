"""Research-only NB10-compatible delayed close-to-close cash/share ledger.

Adjusted-price units, proportional assumed costs, no interest, shorting or leverage.
This supplies no order routing, actual fill guarantees or current trading advice.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradingSpec:
    primary_horizon: int = 3
    initial_capital: float = 100000.
    cost_bps: float = 5.
    edge_buffer_bps: float = 5.
    annual_vol_target: float = .1
    tail_budget: float = .01
    max_weight: float = 1.
    vol_window: int = 20
    cash_return: float = 0.
    shorting: bool = False
    leverage: bool = False

    def __post_init__(self):
        if self.primary_horizon not in range(1,6) or self.vol_window<2 or self.initial_capital<=0:
            raise ValueError('Invalid horizon, risk window or capital')
        if not 0<=self.cost_bps<10000 or self.edge_buffer_bps<0 or not 0<self.max_weight<=1 or self.annual_vol_target<=0 or self.tail_budget<=0:
            raise ValueError('Invalid cost or risk settings')
        if self.cash_return!=0 or self.shorting or self.leverage:raise ValueError('Only zero-interest long/cash is supported')


def decision_plan(forecasts,price,spec=TradingSpec(),horizon=None,model='analog',offset=0,max_forecast_steps=5):
    h=spec.primary_horizon if horizon is None else horizon
    if h not in range(1,6) or h+1>max_forecast_steps or not 0<=offset<h:
        raise ValueError('Holding horizon plus entry delay exceeds declared forecast budget')
    if 'track' in forecasts and not forecasts.track.eq('returns').all():raise ValueError('Trading decisions require actual-return forecasts only')
    f=forecasts[(forecasts.horizon==h)&(forecasts.model==model)].sort_values('date').copy()
    pos=price.index.get_indexer(f.date)
    if len(f)==0 or np.any(pos<0) or not np.all(np.diff(pos)==1):raise ValueError('Need consecutive, matched daily forecast origins')
    f=f.iloc[offset::h].copy();pos=price.index.get_indexer(f.date)
    if np.any(pos+1+h>=len(price)):raise ValueError('Execution outcomes not yet observed')
    f['signal_pos']=pos;f['entry_pos']=pos+1;f['exit_pos']=pos+1+h
    if not np.array_equal(price.index[f.exit_pos],pd.DatetimeIndex(f.label_end)):raise ValueError('Forecast label does not match entry/exit clock')
    vol=np.log(price).diff().rolling(spec.vol_window,min_periods=spec.vol_window).std()*np.sqrt(252)
    f['vol_ann']=vol.reindex(f.date).to_numpy()
    if not np.isfinite(f[['mean','q10','vol_ann']]).all().all() or not f.vol_ann.gt(0).all() or not f.supported.isin([True,False]).all():
        raise ValueError('Invalid causal decision inputs')
    c=spec.cost_bps/1e4
    f['threshold']=np.log((1+c)/(1-c))+spec.edge_buffer_bps/1e4
    f['eligible']=f.supported.astype(bool)&(f['mean']>f.threshold)
    f['tail_loss']=np.maximum(-np.expm1(f.q10),1e-6)
    f['weight']=np.minimum.reduce([np.full(len(f),spec.max_weight),spec.annual_vol_target/f.vol_ann,
                                  spec.tail_budget/f.tail_loss])*f.eligible
    f['reason']=np.where(~f.supported,'unsupported',np.where(~f.eligible,'insufficient edge','long'))
    return f.reset_index(drop=True)


def simulate_plan(plan,price,spec=TradingSpec(),cost_bps=None,window=None):
    c=(spec.cost_bps if cost_bps is None else cost_bps)/1e4
    if not 0<=c<1 or len(plan)==0 or not plan.weight.between(0,1).all():raise ValueError('Invalid costs or allocations')
    start,end=(int(plan.signal_pos.min()),int(plan.exit_pos.max())) if window is None else window
    if start<0 or end>=len(price) or end<=start or (price<=0).any() or not np.isfinite(price).all():raise ValueError('Invalid price path/window')
    if plan.entry_pos.duplicated().any():raise ValueError('Duplicate entry events')
    events={int(r.entry_pos):r for r in plan.itertuples() if r.weight>0}
    cash=spec.initial_capital;units=0.;active=None;rows=[];trades=[]
    for k in range(start,end+1):
        px=float(price.iloc[k]);fees=0.;traded=0.
        # Exit first; a predeclared roll can then enter at the same observed close.
        if active is not None and k==active['exit_pos']:
            proceeds=units*px;fee=proceeds*c;cash+=proceeds-fee;fees+=fee;traded+=proceeds
            active.update(exit_date=price.index[k],exit_price=px,exit_fee=fee,pnl=proceeds-fee-active['budget'],net_return=(proceeds-fee)/active['budget']-1)
            trades.append(active);units=0.;active=None
        if k in events:
            if active is not None:raise ValueError('Overlapping positions')
            r=events[k];budget=cash*r.weight;notional=budget/(1+c);units=notional/px;fee=notional*c
            cash-=budget;fees+=fee;traded+=notional
            active={'signal_date':r.date,'entry_date':price.index[k],'exit_pos':int(r.exit_pos),'entry_price':px,
                    'weight':r.weight,'budget':budget,'units':units,'entry_fee':fee}
        equity=cash+units*px
        if cash < -1e-7 or equity<=0:raise RuntimeError('Invalid cash/equity accounting')
        rows.append({'date':price.index[k],'equity':equity,'cash':cash,'units':units,'exposure':units*px/equity,
                     'fee':fees,'traded_notional':traded})
    if active is not None or units!=0:raise ValueError('Unliquidated final position')
    ledger=pd.DataFrame(rows).set_index('date');ledger['return']=ledger.equity.pct_change().fillna(0)
    ledger['drawdown']=ledger.equity/ledger.equity.cummax()-1
    tr=pd.DataFrame(trades,columns=['signal_date','entry_date','exit_pos','entry_price','weight','budget','units','entry_fee','exit_date','exit_price','exit_fee','pnl','net_return'])
    if not np.isclose(ledger.equity.iloc[-1]-spec.initial_capital,tr.pnl.sum() if len(tr) else 0,atol=1e-7):raise RuntimeError('Ledger PnL does not reconcile')
    return ledger,tr


def trading_metrics(ledger,trades,capital=100000.):
    ret=ledger['return'].iloc[1:];n=len(ret);sd=ret.std(ddof=1)
    return {'total_return':ledger.equity.iloc[-1]/capital-1,'cagr':(ledger.equity.iloc[-1]/capital)**(252/n)-1,
        'annual_vol':sd*np.sqrt(252),'sharpe_zero_cash':np.sqrt(252)*ret.mean()/sd if sd>0 else np.nan,
        'max_drawdown':ledger.drawdown.min(),'mean_exposure':ledger.exposure.iloc[:-1].mean(),
        'trades':len(trades),'win_fraction':(trades.pnl>0).mean() if len(trades) else np.nan,
        'fees':ledger.fee.sum(),'turnover_on_initial':ledger.traded_notional.sum()/capital}
