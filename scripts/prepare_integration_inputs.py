"""Freeze exact NB9/NB10 references and their matching NB3 inputs; no source edits."""
from pathlib import Path
from hashlib import sha256
import json,shutil
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/Users/trush/Documents/Trading/Codex');DEST=ROOT/'inputs/nb_integration';DEST.mkdir(exist_ok=True)
files={}
def copy(source,relative):
    data=source.read_bytes();dest=DEST/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=data:raise ValueError('Frozen integration input differs: '+str(dest))
    dest.write_bytes(data);files[str(relative)]={'sha256':sha256(data).hexdigest(),'source':str(source)}
nb9=json.loads((SOURCE/'outputs/notebook_9/run_manifest.json').read_text())
nb10=json.loads((SOURCE/'outputs/notebook_10/run_manifest.json').read_text())
for name,expected in nb9['input_hashes'].items():
    path=SOURCE/'assets'/name
    assert sha256(path.read_bytes()).hexdigest()==expected
    copy(path,Path('nb3')/name)
for number in [9,10]:
    folder=SOURCE/'outputs'/f'notebook_{number}'
    for path in sorted(folder.iterdir()):
        if path.suffix in ['.csv','.json']:copy(path,Path(f'nb{number}')/path.name)
for name,expected in nb10['input_hashes'].items():assert sha256((DEST/'nb9'/name).read_bytes()).hexdigest()==expected
assert nb10['market_sha256']==files['nb3/market_ohlcv_snapshot.csv']['sha256']
for name in ['Notebook_9_Short_Horizon_Analog_Forecasting.ipynb','Notebook_10_Trading_Decision_Layer.ipynb']:
    copy(SOURCE/name,Path('original_notebooks')/name)
meta={'files':files,'scope':'Immutable NB9/NB10 reference exports and exact NB3 provenance inputs; previously examined SPY history.',
      'source_notebooks':'Read-only originals; no canonical NB9/NB10 changes.',
      'market_sha256':nb10['market_sha256'],'entry_delay_sessions':1,'legacy_horizons':[1,2,3,4,5],
      'library_max_steps':5,'compatible_holding_horizons':[1,2,3,4]}
path=DEST/'reference_manifest.json';data=json.dumps(meta,indent=2).encode()
if path.exists() and path.read_bytes()!=data:raise ValueError('Reference manifest already frozen differently')
path.write_bytes(data);print('Frozen reference files:',len(files),'market hash:',meta['market_sha256'])
