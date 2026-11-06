"""Analytical and replay diagnostics. Not a financial noise estimator."""
from dataclasses import replace
import numpy as np
import pandas as pd
from scipy import signal
from .smoothing import CausalSmoother, smooth


def impulse_response(spec, n=2048):
    if spec.kind in {"kama", "kalman"}:
        raise ValueError("No fixed time-invariant response for adaptive/time-varying estimators")
    engine = CausalSmoother(replace(spec, warmup=1))
    impulse = np.r_[1., np.zeros(n-1)]
    if spec.kind == "identity":
        return impulse
    if spec.kind in {"butter", "bessel"}:
        return signal.sosfilt(engine.sos, impulse)
    if spec.kind in {"ema", "dema", "tema"}:
        e1 = signal.lfilter([spec.alpha], [1., -(1-spec.alpha)], impulse)
        e2 = signal.lfilter([spec.alpha], [1., -(1-spec.alpha)], e1)
        e3 = signal.lfilter([spec.alpha], [1., -(1-spec.alpha)], e2)
        return e1 if spec.kind == "ema" else 2*e1-e2 if spec.kind == "dema" else 3*e1-3*e2+e3
    weights = (np.ones(spec.window)/spec.window if spec.kind == "sma" else
               np.arange(1, spec.window+1)/sum(range(1, spec.window+1)) if spec.kind == "wma"
               else engine.weights)
    if len(weights) > n:
        raise ValueError("Impulse length must cover the FIR window")
    return np.pad(np.asarray(weights)[::-1], (0, n-len(weights)))


def response_metrics(spec):
    kernel = impulse_response(spec)
    w, H = signal.freqz(kernel, worN=2048, fs=1)
    group_delay = -np.gradient(np.unwrap(np.angle(H)), 2*np.pi*w)
    step = np.cumsum(kernel)
    final = kernel.sum()
    outside = np.flatnonzero(np.abs(step-final) > .02*abs(final))
    return {"dc_gain": float(final), "white_noise_gain": float(kernel@kernel),
            "low_frequency_delay": float(np.arange(len(kernel))@kernel/final),
            "overshoot_fraction": float(max(0., step.max()/final-1)),
            "settling_2pct_sessions": int(outside[-1]+1 if len(outside) else 0)}, (w, H, group_delay)


def audit_replay(values, spec, cut=None):
    """Independent prefix/poison/checkpoint checks at a chosen split."""
    x = np.asarray(values, dtype=float)
    cut = len(x)//2 if cut is None else cut
    if not 0 < cut < len(x):
        raise ValueError("Split must lie inside the sample")
    full = smooth(x, spec)
    prefix = smooth(x[:cut], spec)
    np.testing.assert_allclose(full[:cut], prefix, equal_nan=True, rtol=1e-12, atol=1e-12)
    poisoned = x.copy()
    poisoned[cut:] += np.linspace(1e3, 2e3, len(x)-cut)
    np.testing.assert_allclose(full[:cut], smooth(poisoned, spec)[:cut], equal_nan=True)
    engine = CausalSmoother(spec)
    first = engine.update_many(x[:cut])
    checkpoint = engine.checkpoint()
    continuation = CausalSmoother.restore(checkpoint).update_many(x[cut:])
    np.testing.assert_allclose(full, np.r_[first, continuation], equal_nan=True)
    before = engine.checkpoint()
    engine.project(np.repeat(x[cut-1], 5))
    assert before == engine.checkpoint(), "Projection mutated observed state"
    return {"kind": spec.kind, "prefix_invariant": True, "future_poison_invariant": True,
            "checkpoint_equivalent": True, "projection_nonmutating": True}


def latent_scores(observed, latent, specs, start=250):
    """Same latent target/common dates; not raw RMSE between different filters."""
    rows = []
    for name, spec in specs.items():
        estimate = smooth(observed, spec)
        error = estimate[start:]-np.asarray(latent)[start:]
        if not np.isfinite(error).all():
            raise ValueError("Common evaluation start precedes filter maturity")
        rows.append({"filter": name, "n": len(error), "latent_RMSE": float(np.sqrt(np.mean(error**2)))})
    return pd.DataFrame(rows)
