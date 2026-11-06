"""Streaming causal estimators. Every emitted endpoint is immutable.

Parameters are fixed by the caller; no automatic full-sample calibration occurs.
Windows are in observed sessions. Missing/nonfinite inputs fail rather than fill.
"""
from dataclasses import asdict, dataclass
from copy import deepcopy
from functools import lru_cache
import json
import numpy as np
from scipy import signal
from scipy.interpolate import make_smoothing_spline


@lru_cache(maxsize=128)
def spline_endpoint_weights(window, lam):
    """Exact fixed-lambda linear endpoint map, cached for streaming reuse.

    Apply the spline solver to basis vectors; this equals refitting the trailing
    spline for every input window, up to floating point error. Time scale is fixed.
    """
    t = np.arange(window, dtype=float)
    basis = np.eye(window)
    return tuple(float(make_smoothing_spline(t, row, lam=lam)(t[-1])) for row in basis)


@dataclass(frozen=True)
class FilterSpec:
    kind: str = "ema"
    alpha: float = 0.2
    window: int = 21
    degree: int = 2
    cutoff: float = 0.05  # cycles / observed session; Nyquist = 0.5
    order: int = 2
    lam: float = 20.0  # spline time coordinates use session units
    fast_span: int = 2
    slow_span: int = 30
    q_level: float = 1e-5
    q_slope: float = 1e-7
    r: float = 1e-4
    warmup: int = 60

    def __post_init__(self):
        allowed = {"identity", "sma", "wma", "ema", "dema", "tema", "kama",
                   "sg", "local_linear", "spline", "butter", "bessel", "kalman"}
        if self.kind not in allowed:
            raise ValueError(f"Unknown filter: {self.kind}")
        for name in ("window", "degree", "order", "fast_span", "slow_span", "warmup"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if self.window < 2 or self.warmup < 1 or self.order < 1:
            raise ValueError("Require window>=2, warmup>=1, order>=1")
        if not 0 < self.alpha <= 1 or not 0 < self.cutoff < 0.5:
            raise ValueError("alpha in (0,1]; cutoff in (0,0.5) cycles/session")
        if not 0 <= self.degree < self.window:
            raise ValueError("Require 0<=degree<window")
        if self.kind == "spline" and self.window < 5:
            raise ValueError("Smoothing spline requires at least five observations")
        if not (1 <= self.fast_span <= self.slow_span):
            raise ValueError("Require 1<=fast_span<=slow_span")
        if not all(np.isfinite(v) for v in
                   (self.alpha, self.cutoff, self.lam, self.q_level, self.q_slope, self.r)):
            raise ValueError("Filter parameters must be finite")
        if min(self.lam, self.q_level, self.q_slope) < 0 or self.r <= 0:
            raise ValueError("lambda and process variances nonnegative; r positive")

    @property
    def minimum_history(self):
        if self.kind in {"sma", "wma", "sg", "local_linear", "spline"}:
            return max(self.warmup, self.window)
        if self.kind == "kama":
            return max(self.warmup, self.window + 1)
        return self.warmup


class CausalSmoother:
    """One update per observed value; JSON checkpoints contain no future data.

    EMA cascades initialize at the first observation. IIR filters initialize at
    its steady state. KAMA starts there, uses slow-alpha squared until its ER
    window matures, and then uses Kaufman's squared interpolated smoothing rate.
    Kalman uses a local linear trend with fixed Q/R and Joseph covariance update.
    """

    def __init__(self, spec=FilterSpec()):
        self.spec = spec
        self.reset()
        if spec.kind in {"butter", "bessel"}:
            designer = signal.butter if spec.kind == "butter" else signal.bessel
            kwargs = {"norm": "mag"} if spec.kind == "bessel" else {}
            self.sos = designer(spec.order, spec.cutoff, fs=1, output="sos", **kwargs)
        if spec.kind == "sg":
            self.weights = signal.savgol_coeffs(spec.window, spec.degree,
                                               pos=spec.window - 1, use="dot")
        if spec.kind == "spline":
            self.weights = np.array(spline_endpoint_weights(spec.window, spec.lam))
        if spec.kind == "local_linear":
            t = np.arange(spec.window, dtype=float) - (spec.window - 1)
            design = np.column_stack([np.ones(spec.window), t])
            w = np.arange(1, spec.window + 1, dtype=float)
            self.weights = np.linalg.solve(design.T @ (w[:, None]*design),
                                            design.T*w)[0]

    def reset(self):
        self.state = {"n": 0, "buffer": [], "cascade": None, "zi": None,
                      "kama": None, "mean": None, "cov": None, "last_raw": None}

    def update(self, value):
        x = float(value)
        if not np.isfinite(x):
            raise ValueError("Nonfinite observation; choose an explicit missing-data policy")
        s, p = self.state, self.spec
        buf = s["buffer"]
        buf.append(x)
        del buf[:max(0, len(buf) - (p.window + 1))]
        k = p.kind
        if k == "identity":
            y = x
        elif k in {"ema", "dema", "tema"}:
            e = s["cascade"]
            if e is None:
                e = [x, x, x]
            else:
                v = x
                for j in range(3):
                    e[j] += p.alpha * (v - e[j])
                    v = e[j]
            s["cascade"] = e
            y = e[0] if k == "ema" else (2*e[0]-e[1] if k == "dema"
                                        else 3*e[0]-3*e[1]+e[2])
        elif k == "kama":
            er = 0.0
            if len(buf) == p.window + 1:
                volatility = np.abs(np.diff(buf)).sum()
                er = abs(buf[-1] - buf[0])/volatility if volatility else 0.0
            a = (er*(2/(p.fast_span+1)-2/(p.slow_span+1))+2/(p.slow_span+1))**2
            y = x if s["kama"] is None else s["kama"] + a*(x-s["kama"])
            s["kama"] = float(y)
        elif k in {"butter", "bessel"}:
            zi = signal.sosfilt_zi(self.sos)*x if s["zi"] is None else np.array(s["zi"])
            out, zi = signal.sosfilt(self.sos, [x], zi=zi)
            s["zi"], y = zi.tolist(), out[0]
        elif k == "kalman":
            F = np.array([[1., 1.], [0., 1.]])
            if s["mean"] is None:
                m, P = np.array([x, 0.]), np.diag([p.r, p.r])
            else:
                m = F @ np.array(s["mean"])
                P = F @ np.array(s["cov"]) @ F.T + np.diag([p.q_level, p.q_slope])
                gain = P[:, 0]/(P[0, 0]+p.r)
                m += gain*(x-m[0])
                A = np.eye(2) - np.outer(gain, [1., 0.])
                P = A @ P @ A.T + p.r*np.outer(gain, gain)
            s["mean"], s["cov"], y = m.tolist(), P.tolist(), m[0]
        elif len(buf) < p.window:
            y = x  # internal placeholder only; not emitted before maturity
        else:
            trailing = np.asarray(buf[-p.window:])
            if k == "sma":
                y = trailing.mean()
            elif k == "wma":
                y = np.average(trailing, weights=np.arange(1, p.window+1))
            elif k in {"sg", "local_linear", "spline"}:
                y = self.weights @ trailing
        s["n"] += 1
        s["last_raw"] = float(y)
        return float(y) if s["n"] >= p.minimum_history else np.nan

    def update_many(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Expected a finite 1D observed sequence")
        return np.array([self.update(x) for x in values])

    def checkpoint(self):
        return json.loads(json.dumps({"schema_version": 1, "spec": asdict(self.spec),
                                      "state": self.state}, allow_nan=False))

    @classmethod
    def restore(cls, checkpoint):
        if checkpoint.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        obj = cls(FilterSpec(**checkpoint["spec"]))
        obj.state = deepcopy(checkpoint["state"])
        return obj

    def project(self, future_path):
        """Nonmutating filter propagation; nonlinear plug-in is not E[filter]."""
        return self.restore(self.checkpoint()).update_many(future_path)


def smooth(values, spec=FilterSpec()):
    """Fresh replay from the first observation, preserving warm-up NaNs."""
    return CausalSmoother(spec).update_many(values)
