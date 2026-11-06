"""Freeze checksum-verified selected-pipeline references; no notebook execution."""
import argparse
import json
from pathlib import Path
from hashlib import sha256
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--include-ts05',action='store_true')
parser.add_argument('--include-ts06',action='store_true')
args=parser.parse_args()
folder=ROOT/'inputs'/'learning_references';folder.mkdir(exist_ok=True)
metadata_path=folder/'metadata.json'
metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
items=[('TS04_selected','TS04_Classical_Forecasting_and_Selection')]
if args.include_ts05:items.append(('TS05_selected','TS05_Tabular_Machine_Learning'))
if args.include_ts06:items.append(('TS06_selected','TS06_Compact_Neural_Forecasting'))
for label,name in items:
    source=ROOT/'outputs'/name/'historical_forecasts.csv'
    manifest_path=source.parent/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    source_hash=sha256(source.read_bytes()).hexdigest()
    assert manifest['outputs_sha256'][source.name]==source_hash
    frame=pd.read_csv(source)
    frame=frame[frame.model=='selected_pipeline']
    dest=folder/f'{label}.csv'
    data=frame.to_csv(index=False).encode()
    if dest.exists() and dest.read_bytes()!=data:
        raise ValueError('Frozen reference differs; do not silently replace '+str(dest))
    dest.write_bytes(data)
    metadata[label]={'sha256':sha256(data).hexdigest(),'source_forecasts_sha256':source_hash,
        'source_manifest_sha256':sha256(manifest_path.read_bytes()).hexdigest(),
        'note':'Frozen delivered selected pipeline; reruns do not refresh this reference automatically.'}
metadata_path.write_text(json.dumps(metadata,indent=2))
print('Frozen references:',', '.join(metadata))
