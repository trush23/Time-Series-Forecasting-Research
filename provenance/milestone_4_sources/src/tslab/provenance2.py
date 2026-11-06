"""Milestone 2 provenance without altering the accepted milestone 1 writer."""
from importlib.metadata import version
import json
from pathlib import Path
from .reporting import write_manifest


def write_research_manifest(folder, config, source, seed=1731):
    manifest=write_manifest(folder,config,source,seed=seed)
    manifest['scope']='milestone_2_spectral_and_classical_research'
    manifest['versions']['statsmodels']=version('statsmodels')
    manifest['claims']='Exploratory previously examined data; no untouched holdout or trading advantage established'
    (Path(folder)/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
