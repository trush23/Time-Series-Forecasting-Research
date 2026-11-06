"""Local provenance and consistent notebook figures."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import numpy as np
import matplotlib.pyplot as plt
from .data import digest


def configure_plots():
    plt.rcParams.update({"figure.figsize": (10, 4.6), "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.dpi": 120,
                         "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#333333", "#E69F00"])})


def save_figure(fig, folder, name):
    path = Path(folder)/f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    return path


def write_manifest(folder, config, source, tables=(), seed=1731):
    folder = Path(folder)
    config_text = json.dumps(config, sort_keys=True, allow_nan=False)
    root = Path(__file__).resolve().parents[2]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "milestone_1_fixed_configuration_demonstration", "seed": seed,
        "config": config, "config_sha256": sha256(config_text.encode()).hexdigest(),
        "source": source, "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scipy", "matplotlib", "nbformat", "nbconvert"]},
        "source_code_sha256": {str(p.relative_to(root)): digest(p) for p in sorted((root/"src").rglob("*.py"))},
        "outputs_sha256": {p.name: digest(p) for p in sorted(folder.iterdir()) if p.suffix in {".csv", ".png"}},
        "tables": list(tables), "claims": "No selected model, untouched holdout or established trading skill"}
    (folder/"run_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def filter_configs(specs):
    return {name: asdict(spec) for name, spec in specs.items()}
