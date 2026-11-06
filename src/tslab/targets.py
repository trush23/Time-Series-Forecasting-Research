"""Five-observation targets with explicit origin and label availability."""
import numpy as np
import pandas as pd
from .data import validate_series


def validate_horizons(horizons):
    hs = tuple(horizons)
    if not hs or any(isinstance(h, bool) or not isinstance(h, (int, np.integer))
                     or not 1 <= h <= 5 for h in hs) or len(set(hs)) != len(hs):
        raise ValueError("Horizons must be unique integers in 1..5")
    return hs


def make_targets(log_price, smoothed, horizons=range(1, 6), entry_delay=0):
    """Return both equal-priority tracks; retain immature tail rows.

    Smoothed targets are endpoint outputs at origin+h. Returns start at
    origin+entry_delay and end h observations later. entry_delay=1 bridges NB9;
    five such holding periods require six subsequent observations.
    """
    validate_series(log_price)
    validate_series(smoothed, allow_warmup=True)
    if not log_price.index.equals(smoothed.index):
        raise ValueError("Log price and smoothed series indexes must match exactly")
    hs = validate_horizons(horizons)
    if isinstance(entry_delay, bool) or not isinstance(entry_delay, int) or entry_delay < 0:
        raise ValueError("entry_delay must be a nonnegative integer")
    index = log_price.index
    dates = pd.Series(index, index=index)
    frames = []
    for h in hs:
        for track in ("smoothed", "returns"):
            delay = 0 if track == "smoothed" else entry_delay
            target_end = dates.shift(-(h+delay))
            actual = smoothed.shift(-h) if track == "smoothed" else log_price.shift(-(h+delay))-log_price.shift(-delay)
            frames.append(pd.DataFrame({"origin": index, "origin_pos": np.arange(len(index)),
                "track": track, "horizon": h, "entry_delay": delay,
                "target_start": dates.shift(-delay).to_numpy(),
                "target_end": target_end.to_numpy(), "available_at": target_end.to_numpy(),
                "actual": actual.to_numpy(), "mature_in_snapshot": actual.notna().to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def mature_labels(targets, as_of, require_all_horizons=False):
    """Eligibility depends on time AND finite observed labels, never row splitting."""
    as_of = pd.Timestamp(as_of)
    eligible = targets.loc[(targets.available_at <= as_of) &
                           np.isfinite(targets.actual)].copy()
    if require_all_horizons:
        required = targets.groupby("track").horizon.nunique()
        counts = eligible.groupby(["track", "origin"]).horizon.nunique()
        keys = counts[counts.eq(counts.index.get_level_values("track").map(required))].index
        eligible = eligible.set_index(["track", "origin"]).loc[keys].reset_index()
    return eligible
