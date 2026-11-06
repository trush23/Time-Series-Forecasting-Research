"""Explicit online preparation; notebook inference itself uses local pinned weights."""
from pathlib import Path
from hashlib import sha256
from datetime import datetime, timezone
import json
import shutil
import urllib.request
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'inputs/foundation_models'
OUT.mkdir(parents=True, exist_ok=True)
records = []
for size in ['tiny', 'mini']:
    repo = f'amazon/chronos-bolt-{size}'
    info = HfApi().model_info(repo)
    folder = OUT / f'chronos-bolt-{size}'
    cache = snapshot_download(repo, revision=info.sha,
        allow_patterns=['config.json', 'model.safetensors', 'README.md'],
        cache_dir='/private/tmp/ts11-hf')
    folder.mkdir(exist_ok=True)
    for name in ['config.json', 'model.safetensors', 'README.md']:
        shutil.copy2(Path(cache)/name, folder/name)
    records.append({'model_id': repo, 'revision': info.sha,
        'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'model_card_url': f'https://huggingface.co/{repo}/blob/{info.sha}/README.md',
        'license': 'apache-2.0', 'folder': folder.name,
        'training_end_date': None, 'dataset_overlap': 'unknown for every evaluation dataset; not audited zero-shot',
        'files_sha256': {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}})
(OUT/'checkpoint_manifest.json').write_text(json.dumps(records, indent=2))
url = 'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/LICENSE'
with urllib.request.urlopen(url) as response:
    (OUT/'LICENSE-Chronos.txt').write_bytes(response.read())
print(json.dumps(records, indent=2))
