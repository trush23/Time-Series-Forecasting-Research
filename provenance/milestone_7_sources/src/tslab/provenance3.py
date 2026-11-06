"""Milestone 3 metadata, including neural checkpoint digests."""
from importlib.metadata import version
from pathlib import Path
import json
from .reporting import write_manifest
from .data import digest


def write_learning_manifest(folder,config,source,neural=False,seed=1731):
    folder=Path(folder)
    manifest=write_manifest(folder,config,source,seed=seed)
    manifest['scope']='milestone_3_smoothed_input_supervised_forecasting'
    manifest['versions']['scikit-learn']=version('scikit-learn')
    if neural:manifest['versions']['torch']=version('torch')
    manifest['checkpoint_sha256']={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*.pt'))}
    manifest['claims']='Previously examined historical snapshot; no independent test or trading advantage established'
    (folder/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
