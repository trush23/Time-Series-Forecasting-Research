"""Cache only a completed TS09 model stage, bound to exact protocol/source/output bytes."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime,timezone
import json,ast
import pandas as pd
NAMES=('forecasts','inner_forecasts','selections','tuning','classical_audit','tabular_audit','neural_audit',
       'learning_curves','seed_forecasts','scores','paired_uncertainty','block_scores')
DATE_COLUMNS={'origin','training_cutoff','data_available_through','target_start','target_end','available_at',
              'selection_cutoff','outer_first_origin','inner_last_label'}
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
def filename(n):return 'historical_forecasts.csv' if n=='forecasts' else n+'.csv'
def source_hashes(root):return {str(p.relative_to(root)):digest(p) for p in sorted((root/'src').rglob('*.py'))}
def sources_compatible(root,recorded):
    # Additive modules are unused by the recorded dependency set. A package-version
    # assignment may change only when every other initializer AST node is identical.
    for relative,expected in recorded.items():
        current=root/relative
        if digest(current)==expected:continue
        if relative!='src/tslab/__init__.py':return False
        historical=list((root/'provenance').glob('milestone_*_sources/src/tslab/__init__.py'))
        historical+=list((root/'provenance/source_blobs').glob('*.py')) if (root/'provenance/source_blobs').exists() else []
        old=next((p for p in historical if digest(p)==expected),None)
        if old is None:return False
        def normalized(path):
            tree=ast.parse(path.read_text())
            tree.body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets))]
            return ast.dump(tree)
        if normalized(old)!=normalized(current):return False
    return True

def save_cache(root,folder,protocol_hash):
    files=[folder/filename(n) for n in NAMES]+sorted((folder/'checkpoints').rglob('*.pt'))
    meta={'schema':1,'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol_hash,
          'source_sha256':source_hashes(root),'files_sha256':{str(p.relative_to(folder)):digest(p) for p in files}}
    (folder/'completed_model_stage.json').write_text(json.dumps(meta,indent=2))
    return meta

def load_cache(root,folder,protocol_hash):
    path=folder/'completed_model_stage.json'
    if not path.exists():return None
    meta=json.loads(path.read_text())
    if meta.get('schema')!=1 or meta['protocol_sha256']!=protocol_hash or not sources_compatible(root,meta['source_sha256']):
        raise ValueError('Completed TS09 stage differs from current protocol/code; set RECOMPUTE=True to run anew')
    for name,checksum in meta['files_sha256'].items():
        if digest(folder/name)!=checksum:raise ValueError('Completed TS09 result bytes changed; set RECOMPUTE=True: '+name)
    result={}
    for name in NAMES:
        frame=pd.read_csv(folder/filename(name))
        for col in DATE_COLUMNS.intersection(frame.columns):frame[col]=pd.to_datetime(frame[col])
        result[name]=frame
    return result
