"""Retrospective FFT decompositions and separately archived causal endpoints.

FFT bins use observed-session coordinates. A frequency in cycles/session is not
a calendar-time period. No estimated cycle is assumed to persist automatically.
"""
from dataclasses import dataclass, asdict
import json
import numpy as np


@dataclass(frozen=True)
class SpectralSpec:
    window: int = 128
    low: float = 0.0
    high: float = 0.05
    detrend: bool = True

    def __post_init__(self):
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 8:
            raise ValueError("FFT window must be an integer >=8")
        if not np.isfinite([self.low, self.high]).all() or not 0 <= self.low < self.high <= .5:
            raise ValueError("Require 0<=low<high<=Nyquist 0.5 cycles/session")


def fft_decompose(values, low=0., high=.05, detrend=True):
    """Whole-vector, retrospective decomposition; components sum to input.

    Trend is OLS linear (or a mean). The band excludes DC when low>0. High-pass
    residual is defined algebraically; no tapered reconstruction is mis-scaled.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("Need >=8 finite 1D observations")
    SpectralSpec(window=len(x), low=low, high=high, detrend=detrend)
    t = np.arange(len(x), dtype=float)
    trend = np.polyval(np.polyfit(t, x, 1), t) if detrend else np.full(len(x), x.mean())
    residual = x-trend
    frequencies = np.fft.rfftfreq(len(x))
    spectrum = np.fft.rfft(residual)
    mask = (frequencies >= low) & (frequencies <= high)
    band = np.fft.irfft(spectrum*mask, n=len(x))
    return {"trend": trend, "band": band, "remainder": x-trend-band,
            "reconstructed": trend+band, "frequencies": frequencies,
            "spectrum": spectrum, "mask": mask}


class SpectralEndpoint:
    """Replay one trailing FFT per endpoint. Previously issued values never revise."""
    def __init__(self, spec=SpectralSpec()):
        self.spec = spec
        self.buffer = []
        self.n = 0

    def update(self, value):
        if not np.isfinite(value):
            raise ValueError("Nonfinite observation")
        self.buffer.append(float(value))
        self.buffer = self.buffer[-self.spec.window:]
        self.n += 1
        if len(self.buffer) < self.spec.window:
            return {"trend":np.nan, "band":np.nan, "reconstructed":np.nan}
        fit = fft_decompose(self.buffer, self.spec.low, self.spec.high, self.spec.detrend)
        return {k:float(fit[k][-1]) for k in ["trend","band","reconstructed"]}

    def checkpoint(self):
        return json.loads(json.dumps({"schema_version":1,"spec":asdict(self.spec),
                                      "buffer":self.buffer,"n":self.n}))

    @classmethod
    def restore(cls, state):
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        obj = cls(SpectralSpec(**state["spec"]))
        obj.buffer, obj.n = list(state["buffer"]), state["n"]
        return obj

    def project(self, values):
        clone = self.restore(self.checkpoint())
        return [clone.update(value) for value in values]


def spectral_replay(values, spec=SpectralSpec()):
    engine = SpectralEndpoint(spec)
    rows = [engine.update(x) for x in values]
    return {k:np.array([row[k] for row in rows]) for k in rows[0]}


def harmonic_forecast(values, horizon=5, window=128, harmonics=2, low=.01, high=.15):
    """Training-window trend + Fourier extrapolation. NOT an FFT smoother forecast.

    Rank eligible bins by training amplitude; joint OLS refit avoids assuming
    detrended finite-window Fourier coefficients are orthogonal to the trend.
    No taper is used in reconstruction. Return the chosen bins for audit.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or len(x) < window:
        raise ValueError("Insufficient finite history")
    if not isinstance(horizon, int) or not 1 <= horizon <= 5:
        raise ValueError("horizon must be an integer in 1..5")
    if not isinstance(harmonics, int) or harmonics < 0:
        raise ValueError("harmonics must be a nonnegative integer")
    fit = fft_decompose(x[-window:], low, high, True)
    eligible = np.flatnonzero(fit["mask"] & (fit["frequencies"] > 0) & (fit["frequencies"] < .5))
    if harmonics > len(eligible):
        raise ValueError("Not enough eligible frequency bins")
    ranked = sorted(eligible, key=lambda j:(-abs(fit["spectrum"][j]), j))[:harmonics]
    frequencies = fit["frequencies"][ranked]

    def design(t):
        columns = [np.ones(len(t)), t/(window-1)]
        for f in frequencies:
            columns.extend([np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)])
        return np.column_stack(columns)

    training = np.arange(window, dtype=float)
    beta = np.linalg.lstsq(design(training), x[-window:], rcond=None)[0]
    forecast = design(np.arange(window, window+horizon, dtype=float)) @ beta
    return forecast, {"frequencies":frequencies.tolist(),"periods_sessions":(1/frequencies).tolist(),
                      "fitted_endpoint_gap":float((design(training[-1:])@beta)[0]-x[-1]),
                      "window":window,"harmonics":harmonics,"frequency_selection":"past amplitude only"}
