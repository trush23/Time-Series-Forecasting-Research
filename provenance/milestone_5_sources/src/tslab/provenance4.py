"""Milestone 4 metadata for explicitly adapted architecture demonstrations."""
from pathlib import Path
import json
from .provenance3 import write_learning_manifest


def write_advanced_manifest(folder,config,source,seed=1731):
    manifest=write_learning_manifest(folder,config,source,neural=True,seed=seed)
    manifest['scope']='milestone_4_adapted_architecture_demonstrations'
    manifest['architecture_claim']='Educational feature-to-target adapters; not original published model reproductions'
    (Path(folder)/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
