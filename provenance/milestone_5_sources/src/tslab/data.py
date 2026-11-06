"""Explicit, offline snapshots. No download or synthetic fallback."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
import pandas as pd


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_series(series, allow_warmup=False):
    if not isinstance(series, pd.Series) or not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Expected a Series indexed by actual observed session timestamps")
    if len(series) == 0 or series.index.hasnans or series.index.has_duplicates:
        raise ValueError("Empty, missing or duplicated session timestamps")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Session timestamps must increase strictly")
    values = series.to_numpy(dtype=float)
    if allow_warmup:
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite) or not np.isnan(values[:finite[0]]).all() or not np.isfinite(values[finite[0]:]).all():
            raise ValueError("Only a leading warm-up NaN prefix is allowed")
    elif not np.isfinite(values).all():
        raise ValueError("Nonfinite values; no implicit fill policy")
    return series


def load_snapshot(path, metadata_path):
    """Require an explicit digest and adjustment/source declarations."""
    path, metadata_path = Path(path), Path(metadata_path)
    meta = json.loads(metadata_path.read_text())
    for key in ("sha256", "ticker", "source", "synthetic", "adjustment", "calendar"):
        if key not in meta:
            raise ValueError(f"Missing metadata field {key}")
    if digest(path) != meta["sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    close = validate_series(frame["Close"].astype(float))
    if (close <= 0).any():
        raise ValueError("Close prices must be positive")
    return np.log(close).rename("log_price"), meta


def synthetic_signal(n=800, seed=1731, noise=0.012):
    """Known latent trend/cycle plus IID measurement noise; not market data.

    Weekday timestamps are a synthetic observation schedule, not an exchange
    calendar. Horizons always advance through the supplied observed index.
    """
    if n < 100 or noise < 0:
        raise ValueError("Require n>=100 and noise>=0")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    latent = 4.5 + 0.0006*t + 0.035*np.sin(2*np.pi*t/80) + 0.025*(t >= n//2)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"latent": latent, "observed": latent+rng.normal(0, noise, n)}, index=index)
