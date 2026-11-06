"""Retain every source file used by delivered notebooks before the stable ZIP changes."""
from pathlib import Path
from hashlib import sha256
import json,zipfile
import nbformat


def archive_execution_sources(root):
    count=0
    for path in sorted((root/'notebooks').glob('TS*.ipynb')):
        nb=nbformat.read(path,as_version=4);generation=nb.metadata.get('milestone',1)
        manifest=json.loads((root/'outputs'/path.stem/'run_manifest.json').read_text())
        for relative,expected in manifest['source_code_sha256'].items():
            target=root/'provenance'/f'milestone_{generation}_sources'/relative
            if target.exists() and sha256(target.read_bytes()).hexdigest()==expected:continue
            if target.exists():target=root/'provenance/source_blobs'/f'{expected}.py'
            if target.exists():
                assert sha256(target.read_bytes()).hexdigest()==expected
                continue
            current=root/relative
            if sha256(current.read_bytes()).hexdigest()==expected:data=current.read_bytes()
            else:
                legacy=root.parent/f'Time_Series_Forecasting_Milestone_{generation}.zip'
                with zipfile.ZipFile(legacy) as bundle:data=bundle.read('forecasting_library/'+relative)
                if sha256(data).hexdigest()!=expected:raise ValueError('Historical source digest mismatch')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);count+=1
    return count

if __name__=='__main__':print('Archived execution source files:',archive_execution_sources(Path(__file__).resolve().parents[1]))
