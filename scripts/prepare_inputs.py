"""Freeze the existing project demonstration snapshot; no network access."""
from pathlib import Path
from hashlib import sha256
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT.parent/"notebook_3"/"assets"
manifest = json.loads((source/"run_manifest.json").read_text())
provenance = manifest["provenance"]
if provenance["synthetic"] or provenance["ticker"] != "SPY":
    raise ValueError("Expected the established real SPY demonstration snapshot")
destination = ROOT/"inputs"
destination.mkdir(exist_ok=True)
snapshot = destination/"SPY_adjusted_demo.csv"
if snapshot.exists() and snapshot.read_bytes() != (source/"market_ohlcv_snapshot.csv").read_bytes():
    raise ValueError("Existing bundled snapshot differs; do not overwrite it silently")
shutil.copyfile(source/"market_ohlcv_snapshot.csv", snapshot)
meta = {"ticker": "SPY", "source": provenance["source"], "synthetic": False,
        "adjustment": provenance["adjustment"], "calendar": "observed Yahoo SPY session index",
        "first": provenance["first"], "last": provenance["last"], "rows": provenance["rows"],
        "retrieved_utc": provenance["retrieved_utc"],
        "sha256": sha256(snapshot.read_bytes()).hexdigest(),
        "upstream_manifest_sha256": sha256((source/"run_manifest.json").read_bytes()).hexdigest(),
        "note": "Frozen local project snapshot; previously examined data, not an untouched holdout. "
                "Byte digest is computed from the exported CSV, not the upstream internal dataframe hash. "
                "Adjusted history is not executable historical quotes or point-in-time revision history."}
(destination/"SPY_adjusted_demo.metadata.json").write_text(json.dumps(meta, indent=2))
print("Frozen SPY demo:", meta["rows"], "sessions; SHA256", meta["sha256"])
