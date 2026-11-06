"""Package source, executed notebooks and checked outputs without build caches."""
from hashlib import sha256
from pathlib import Path
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
from archive_execution_sources import archive_execution_sources
archive_execution_sources(ROOT)
excluded = {"__pycache__", ".pytest_cache", "build"}


def deliverable(path):
    parts = path.relative_to(ROOT).parts
    return (path.is_file() and not any(p in excluded or p.endswith(".egg-info") for p in parts)
            and path.name != "checksums.json")


files = sorted(p for p in ROOT.rglob("*") if deliverable(p))
checksums = {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT/"checksums.json").write_text(json.dumps(checksums,indent=2))
files.append(ROOT/"checksums.json")
archive = ROOT.parent/"Time_Series_Forecasting.zip"
design = ROOT.parent/"forecasting_library_research"/"DESIGN_RESEARCH.md"
with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
    for p in files:
        bundle.write(p, "forecasting_library/"+str(p.relative_to(ROOT)))
    bundle.write(design,"forecasting_library_research/DESIGN_RESEARCH.md")
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    for filename, checksum in checksums.items():
        assert sha256(bundle.read("forecasting_library/"+filename)).hexdigest() == checksum
print(json.dumps({"archive":str(archive),"files":len(files)+1,
                  "bytes":archive.stat().st_size,"sha256":sha256(archive.read_bytes()).hexdigest()},indent=2))
