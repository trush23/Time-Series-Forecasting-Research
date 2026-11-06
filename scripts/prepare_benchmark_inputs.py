"""Freeze local caches and bundled public climate data; no network/fill policy."""
from pathlib import Path
from hashlib import sha256
from importlib.metadata import version
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import nbformat
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'inputs'/'cross_dataset';DEST.mkdir(exist_ok=True)
CACHE=Path('/Users/trush/Documents/Trading/Codex/data')
items={'SPY':'7588b324a5eb162a702e','MCHP':'16413e5be6ba98e8ac78',
       'MRVL':'966ef2f114ac9795eeb3','TXG':'1cb3a2f48ce4c1bf3b7d',
       'GOOGL':'c9b2d35712a5306f1c00','NVDA':'ohlcv_8d7e3dcd9d4343d5'}
catalogue=[]
producer=Path('/Users/trush/Documents/Trading/Codex/NB1_Supplement_End_of_Day_Operator_Dashboard.ipynb')
producer_nb=nbformat.read(producer,as_version=4)
loader=next(c.source for c in producer_nb.cells if c.cell_type=='code' and 'def adjusted(' in c.source)
assert "factor=z['Adj Close']/z.Close" in loader and 'z=adjusted(z);z.to_csv(p)' in loader
producer_evidence={'notebook_sha256':sha256(producer.read_bytes()).hexdigest(),
    'loader_sha256':sha256(loader.encode()).hexdigest(),'adjustment':'Download auto_adjust=False then scale OHLC by Adj Close / Close before caching',
    'loader_source':loader}

def frozen(path,data):
    if path.exists() and path.read_bytes()!=data:raise ValueError('Do not replace a frozen benchmark input: '+str(path))
    path.write_bytes(data)
def save(identifier,frame,meta):
    path=DEST/f'{identifier}.csv';frozen(path,frame.to_csv(index=False).encode())
    meta.update(dataset_id=identifier,sha256=sha256(path.read_bytes()).hexdigest(),rows=len(frame),
                first=str(frame.Date.iloc[0].date()),last=str(frame.Date.iloc[-1].date()))
    frozen(DEST/f'{identifier}.metadata.json',json.dumps(meta,indent=2).encode())
    catalogue.append(meta)
for ticker,name in items.items():
    source=CACHE/f'{name}.csv';m=json.loads((CACHE/f'{name}.json').read_text())
    assert m['ticker']==ticker and sha256(source.read_bytes()).hexdigest()==m['sha256']
    frame=pd.read_csv(source,parse_dates=['Date'])
    frame=frame[frame.Date.between('2021-01-04','2026-10-01')].copy()
    assert frame.Date.is_monotonic_increasing and not frame.Date.duplicated().any()
    assert np.isfinite(frame.drop(columns='Date').to_numpy()).all() and (frame.Close>0).all()
    save(ticker,pd.DataFrame({'Date':frame.Date,'value':np.log(frame.Close)}),
         {'group':'stocks','transform':'log adjusted Close','units':'log units','frequency':'observed trading sessions',
          'source':'Yahoo via yfinance; frozen existing local cache','source_cache_name':source.name,
          'source_csv_sha256':m['sha256'],'source_metadata':m,
          'adjustment':m.get('adjustment','OHLC multiplied by provider Adj Close / Close before caching'),
          'producer_evidence_sha256':sha256(json.dumps(producer_evidence,sort_keys=True).encode()).hexdigest() if ticker!='NVDA' else None,
          'exposure':'Previously used in other project notebooks; not a pristine asset holdout.'})
frozen(DEST/'cache_adjustment_provenance.json',json.dumps(producer_evidence,indent=2).encode())
d=sm.datasets.elnino.load_pandas().data
values=d.drop(columns='YEAR').to_numpy().reshape(-1)
dates=pd.date_range('1950-01-31',periods=len(values),freq='ME')
save('ELNINO',pd.DataFrame({'Date':dates,'value':values}),{'group':'nonfinancial','transform':'identity, Celsius level',
     'units':'degrees Celsius','frequency':'monthly','source':'statsmodels bundled NOAA ERSST.V3B Nino 1+2 monthly series',
     'source_url':'https://www.statsmodels.org/stable/datasets/generated/elnino.html','statsmodels_version':version('statsmodels'),
     'exposure':'Public teaching dataset; no pristine data-independence claim.'})
x=sm.datasets.co2.load_pandas().data.co2
monthly=x.resample('ME').agg(['mean','count']);monthly=monthly.loc['1965-01-01':]
assert monthly['mean'].notna().all() and monthly['count'].ge(1).all()
frozen(DEST/'CO2_monthly_observation_counts.csv',monthly[['count']].to_csv(index_label='Date').encode())
save('CO2',monthly['mean'].rename('value').rename_axis('Date').reset_index(),
     {'group':'nonfinancial','transform':'identity, concentration level','units':'ppm','frequency':'monthly',
      'source':'statsmodels bundled Mauna Loa weekly CO2; mean of finite observations within each month',
      'source_url':'https://www.statsmodels.org/stable/datasets/generated/co2.html','statsmodels_version':version('statsmodels'),
      'aggregation':'1965 onward contiguous monthly means; no interpolation or future fill; available at month end',
      'minimum_weekly_observations_per_month':int(monthly['count'].min()),
      'exposure':'Public teaching dataset; no pristine data-independence claim.'})
frozen(DEST/'catalogue.json',json.dumps(catalogue,indent=2).encode())
print('Frozen:',[(m['dataset_id'],m['rows']) for m in catalogue])
