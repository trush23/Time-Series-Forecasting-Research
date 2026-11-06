"""Minimal origin-by-origin benchmarks. No fitting or hyperparameter selection.

Forecast routines receive only prefixes. Labels are attached after forecasts.
"""
import numpy as np
import pandas as pd
from .data import validate_series
from .smoothing import CausalSmoother
from .targets import make_targets, validate_horizons


def null_predictions(log_history, smooth_history, smoother, horizons=range(1, 6), drift_window=60):
    """Same-origin targets only; use a separate experiment for delayed execution.

    Returns use smooth_slope as the smoothed-input candidate; zero and historical
    raw drift are benchmark comparators. Smoothed forecasts include full filter
    state propagation along unchanged/drift log-price paths. For KAMA these are
    plug-in paths, not conditional expectations; Kalman gains remain time varying.
    """
    hs = validate_horizons(horizons)
    x, s = np.asarray(log_history), np.asarray(smooth_history)
    if len(x) != len(s) or len(x) == 0 or not np.isfinite(x).all() or not np.isfinite(s[-1]):
        raise ValueError("Need aligned histories and a mature current smoother output")
    max_h = max(hs)
    drift = np.diff(x[-(drift_window+1):]).mean() if len(x) > 1 else 0.
    slope = s[-1]-s[-2] if len(s) > 1 and np.isfinite(s[-2]) else 0.
    no_change = smoother.project(np.repeat(x[-1], max_h))
    drift_path = smoother.project(x[-1] + drift*np.arange(1, max_h+1))
    rows = []
    for h in hs:
        for model, value in {"persistence": s[-1], "smooth_slope": s[-1]+h*slope,
                             "filter_nochange": no_change[h-1], "filter_drift": drift_path[h-1]}.items():
            rows.append({"track": "smoothed", "horizon": h, "model": model, "prediction": value})
        for model, value in {"zero": 0., "raw_drift": h*drift, "smooth_slope": h*slope}.items():
            rows.append({"track": "returns", "horizon": h, "model": model, "prediction": value})
    return pd.DataFrame(rows)


def benchmark(log_price, spec, start=250, step=5, horizons=range(1, 6), filter_id=None):
    """Chronological replay; preserve immature latest forecasts in the export."""
    validate_series(log_price)
    hs = validate_horizons(horizons)
    if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (start, step))
            or start < spec.minimum_history or start >= len(log_price) or step < 1):
        raise ValueError("Require a mature start within the sample and positive step")
    engine = CausalSmoother(spec)
    values, rows = [], []
    x = log_price.to_numpy()
    for t, value in enumerate(x):
        values.append(engine.update(value))
        if t >= start and ((t-start) % step == 0 or t == len(x)-1):
            forecasts = null_predictions(x[:t+1], values, engine, hs)
            forecasts["origin"] = log_price.index[t]
            forecasts["training_cutoff"] = log_price.index[t]
            forecasts["data_available_through"] = log_price.index[t]
            rows.append(forecasts)
    s = pd.Series(values, index=log_price.index)
    targets = make_targets(log_price, s, hs)
    result = pd.concat(rows, ignore_index=True).merge(targets, on=["origin", "track", "horizon"],
                                                       validate="many_to_one")
    result["filter_id"] = filter_id or spec.kind
    result["units"] = np.where(result.track == "returns", "log_return", "log_level")
    result["error"] = result.prediction-result.actual
    result["status"] = np.where(result.actual.notna(), "scored", "immature")
    return result, s


def scorecard(records):
    """Same-target paired baseline skill; NaN skill when baseline loss is zero."""
    scored = records.loc[records.status == "scored"].copy()
    keys = ["filter_id", "track", "horizon"]
    out = []
    for group, part in scored.groupby(keys):
        baseline = "filter_nochange" if group[1] == "smoothed" else "zero"
        pivot = part.pivot(index="origin", columns="model", values="error")
        if baseline not in pivot or pivot.isna().any().any():
            raise ValueError("Incomplete matched-origin comparison")
        base_mae, base_mse = pivot[baseline].abs().mean(), (pivot[baseline]**2).mean()
        for model in pivot:
            e = pivot[model]
            mae, mse = e.abs().mean(), (e**2).mean()
            out.append(dict(zip(keys, group), model=model, n=len(e), MAE=mae,
                            RMSE=np.sqrt(mse), baseline=baseline,
                            MAE_skill=1-mae/base_mae if base_mae > 0 else np.nan,
                            MSE_skill=1-mse/base_mse if base_mse > 0 else np.nan))
    return pd.DataFrame(out)


def paired_block_difference(records, track, model, baseline, horizon=5,
                            block=10, draws=1000, seed=1731):
    """Exploratory paired MAE difference, resampling scheduled origins in blocks.

    block counts forecast origins (not sessions). Conditional on fixed design;
    no correction for selection or nonstationarity. Use a single filter at a time.
    """
    sub = records.loc[(records.track == track) & (records.horizon == horizon) &
                      (records.status == "scored")]
    if sub.filter_id.nunique() != 1:
        raise ValueError("Select exactly one filter before paired inference")
    errors = sub.pivot(index="origin", columns="model", values="error").sort_index()
    delta = (errors[model].abs()-errors[baseline].abs()).to_numpy()
    if not 1 <= block <= len(delta) or draws < 2:
        raise ValueError("Invalid block length or resample count")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(delta)-block+1, size=(draws, int(np.ceil(len(delta)/block))))
    indexes = (starts[..., None]+np.arange(block)).reshape(draws, -1)[:, :len(delta)]
    means = delta[indexes].mean(axis=1)
    return {"mean_MAE_difference": float(delta.mean()),
            "lower95": float(np.quantile(means, .025)), "upper95": float(np.quantile(means, .975)),
            "n_origins": len(delta), "block_origins": block, "draws": draws}
