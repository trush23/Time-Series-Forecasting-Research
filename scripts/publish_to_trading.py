"""Replace the stable Trading/Codex delivery after checksum verification.

User-authored changes stop publication unless the staged package contains those
exact changed bytes, preserving the edit. Extra user files are retained. No numbered delivery is created.
"""
from pathlib import Path
from hashlib import sha256
import json,shutil,tempfile,zipfile,os
ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT.parent/'Time_Series_Forecasting.zip'
PARENT=Path('/Users/trush/Documents/Trading/Codex')
TARGET=PARENT/'Time_Series_Forecasting'
ZIP_TARGET=PARENT/'Time_Series_Forecasting.zip'
def digest(p):return sha256(Path(p).read_bytes()).hexdigest()
with zipfile.ZipFile(ARCHIVE) as bundle:
    if bundle.testzip() is not None:raise ValueError('Delivery ZIP is corrupt')
    for name in bundle.namelist():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
    with tempfile.TemporaryDirectory(prefix='.tsforecast-stage-',dir=PARENT) as temp:
        stage=Path(temp)/'delivery';stage.mkdir();bundle.extractall(stage)
        library=stage/'forecasting_library'
        checks=json.loads((library/'checksums.json').read_text())
        for name,expected in checks.items():
            if digest(library/name)!=expected:raise ValueError('Staged checksum mismatch: '+name)
        old_owned=set()
        old_checks=TARGET/'forecasting_library/checksums.json'
        if TARGET.exists():
            if not old_checks.is_file():raise ValueError('Existing stable folder is not a managed delivery; review it first')
            prior=json.loads(old_checks.read_text())
            for name,expected in prior.items():
                f=TARGET/'forecasting_library'/name
                if not f.is_file() or (digest(f)!=expected and checks.get(name)!=digest(f)):
                    raise ValueError('User changes or deletion found; review before overwrite: '+str(f))
                old_owned.add(Path('forecasting_library')/name)
            old_owned.add(Path('forecasting_library/checksums.json'))
            old_owned.add(Path('forecasting_library_research/DESIGN_RESEARCH.md'))
            for f in TARGET.rglob('*'):
                if f.is_symlink():raise ValueError('Review symlink before publishing: '+str(f))
                if f.is_file() and f.relative_to(TARGET) not in old_owned:
                    relative=f.relative_to(TARGET);dest=stage/relative
                    if dest.exists() and dest.read_bytes()!=f.read_bytes():raise ValueError('New managed path conflicts with user file: '+str(relative))
                    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        # Keep rollback only during replacement; no accumulating copies.
        previous=Path(temp)/'previous'
        if TARGET.exists():TARGET.rename(previous)
        try:stage.rename(TARGET)
        except Exception:
            if previous.exists():previous.rename(TARGET)
            raise
        zip_stage=Path(temp)/'Time_Series_Forecasting.zip'
        shutil.copy2(ARCHIVE,zip_stage);os.replace(zip_stage,ZIP_TARGET)
        for name,expected in checks.items():assert digest(TARGET/'forecasting_library'/name)==expected
        assert digest(ZIP_TARGET)==digest(ARCHIVE)
print(json.dumps({'folder':str(TARGET),'archive':str(ZIP_TARGET),'archive_sha256':digest(ZIP_TARGET),
    'managed_files_verified':len(checks),'policy':'Update these same paths on subsequent deliveries'},indent=2))
