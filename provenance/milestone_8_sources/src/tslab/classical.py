"""Native statistical smoothed-series models and an explicit actual-return bridge.

All training inputs are supplied prefixes. Fitting failures are exceptions to the
caller; the evaluation harness owns the documented operational fallback policy.
"""
from dataclasses import dataclass
import time
import warnings
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tools.sm_exceptions import ConvergenceWarning


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    lags: int = 1
    order: tuple = (1,1,1)

    def __post_init__(self):
        if self.family not in {"ar_diff","arma_diff","arima","ses","holt","theta","local_level","local_trend"}:
            raise ValueError("Unknown classical model family")
        if not isinstance(self.lags,int) or self.lags < 1:
            raise ValueError("lags must be a positive integer")
        if len(self.order) != 3 or any(not isinstance(v,int) or v < 0 for v in self.order):
            raise ValueError("ARIMA order must contain three nonnegative integers")


def default_models():
    return [ModelSpec("AR1_changes","ar_diff",lags=1), ModelSpec("AR5_changes","ar_diff",lags=5),
            ModelSpec("MA1_changes","arma_diff",order=(0,0,1)),
            ModelSpec("ARMA11_changes","arma_diff",order=(1,0,1)),
            ModelSpec("ARIMA111","arima",order=(1,1,1)),
            ModelSpec("SES_ETS_ANN","ses"), ModelSpec("Holt_damped","holt"),
            ModelSpec("Theta2","theta"), ModelSpec("Local_level","local_level"),
            ModelSpec("Local_linear_trend","local_trend")]


def native_forecast(history, spec, horizon=5, train_window=252):
    """Forecast causal smoothed levels. Residuals are in log units.

    Normalization is computed only on this training prefix. Nonconverged optimizers
    fail explicitly. AR/MA/ARMA on first differences integrate back to the last level.
    Constant paths return their constant rather than invoke an ill-posed optimizer.
    """
    values = np.asarray(history, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 40:
        raise ValueError("Need at least 40 finite smoothed observations")
    if not isinstance(horizon,int) or not 1 <= horizon <= 5 or train_window < 40:
        raise ValueError("Require horizon 1..5 and train_window>=40")
    y = values[-train_window:]
    center, scale = y[-1], np.std(y)
    started = time.perf_counter()
    if scale < 1e-12:
        return np.repeat(y[-1],horizon), {"converged":True,"constant_path":True,
            "warnings":"", "n_train":len(y), "runtime_seconds":0., "residuals":np.zeros(len(y))}
    z = (y-center)/scale
    diff_scale = max(np.std(np.diff(y)),1e-12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if spec.family == "ar_diff":
            d = np.diff(y)/diff_scale
            fit = AutoReg(d, lags=spec.lags, trend="c", old_names=False).fit()
            pred = fit.predict(start=len(d),end=len(d)+horizon-1)
            forecast = y[-1]+np.cumsum(pred)*diff_scale
            residuals = np.asarray(fit.resid)*diff_scale
        elif spec.family == "arma_diff":
            d = np.diff(y)/diff_scale
            fit = ARIMA(d, order=spec.order, trend="c").fit(method_kwargs={"maxiter":100})
            forecast = y[-1]+np.cumsum(fit.forecast(horizon))*diff_scale
            residuals = np.asarray(fit.resid)*diff_scale
        elif spec.family == "arima":
            # Linear trend in an I(1) level gives a drift in first differences.
            trend = "t" if spec.order[1]==1 else "c" if spec.order[1]==0 else "n"
            fit = ARIMA(z,order=spec.order,trend=trend).fit(method_kwargs={"maxiter":100})
            forecast = center+scale*np.asarray(fit.forecast(horizon))
            residuals = np.asarray(fit.resid)*scale
        elif spec.family == "ses":
            fit = SimpleExpSmoothing(z,initialization_method="estimated").fit()
            forecast = center+scale*np.asarray(fit.forecast(horizon))
            residuals = np.asarray(fit.resid)*scale
        elif spec.family == "holt":
            fit = Holt(z,damped_trend=True,initialization_method="estimated").fit()
            forecast = center+scale*np.asarray(fit.forecast(horizon))
            residuals = np.asarray(fit.resid)*scale
        elif spec.family == "theta":
            fit = ThetaModel(z,period=1,deseasonalize=False).fit()
            forecast = center+scale*np.asarray(fit.forecast(horizon,theta=2))
            residuals = np.array([])  # Theta API has no equivalent innovation series here
        else:
            level = "local level" if spec.family == "local_level" else "local linear trend"
            fit = UnobservedComponents(z,level=level).fit(disp=False,maxiter=100)
            forecast = center+scale*np.asarray(fit.forecast(horizon))
            residuals = np.asarray(fit.resid)*scale
    optimizer = getattr(fit,"mle_retvals", {})
    converged = (bool(optimizer.get("converged",optimizer.get("success",True)))
                 and not any(issubclass(w.category,ConvergenceWarning) for w in caught))
    warning_text = " | ".join(str(w.message) for w in caught)
    if not converged:
        raise RuntimeError("Optimizer did not converge: "+warning_text)
    forecast = np.asarray(forecast,dtype=float)
    if forecast.shape != (horizon,) or not np.isfinite(forecast).all():
        raise RuntimeError("Invalid native forecast")
    return forecast, {"converged":True,"constant_path":False,"warnings":warning_text,
        "n_train":len(y),"runtime_seconds":time.perf_counter()-started,"residuals":residuals}


def fit_return_bridge(log_history, smooth_history, train_window=252):
    """OLS r_(j+1) = a + b*(s_j-s_(j-1)); only matured daily labels.

    The predictor is a smoothed input; actual returns are supervised labels.
    Latest training pair uses feature at t-1 and the observed return ending at t.
    No inversion of smoothing or substitution of smoothed changes for raw labels.
    """
    x, s = np.asarray(log_history,dtype=float), np.asarray(smooth_history,dtype=float)
    if len(x) != len(s) or x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Need aligned finite log-price history")
    j = np.arange(max(1,len(x)-train_window),len(x)-1)
    feature = s[j]-s[j-1]
    target = x[j+1]-x[j]
    valid = np.isfinite(feature)
    feature, target, j = feature[valid], target[valid], j[valid]
    if len(feature) < 30:
        raise ValueError("Insufficient mature return-calibration pairs")
    u, v = feature-feature.mean(), target-target.mean()
    denom = u@u
    beta = float(u@v/denom) if denom > 1e-20 else 0.
    intercept = float(target.mean()-beta*feature.mean())
    return {"intercept":intercept,"beta":beta,"n_pairs":len(j),
            "last_feature_pos":int(j[-1]),"last_label_pos":int(j[-1]+1)}


def bridge_forecast(native_path, smooth_history, bridge):
    """Cumulative actual-return forecasts using projected lagged smoothed slopes.

    At h=1 all native families share the currently observed slope and therefore
    the same return forecast. Families differ only at longer horizons. This is a
    restricted two-stage model, not native ARIMA forecasts of actual returns.
    """
    s = np.asarray(smooth_history,dtype=float)
    path = np.asarray(native_path,dtype=float)
    if len(s)<2 or not np.isfinite(s[-2:]).all() or not np.isfinite(path).all():
        raise ValueError("Need mature current and preceding endpoints")
    slopes = np.r_[s[-1]-s[-2],np.diff(np.r_[s[-1],path[:-1]])]
    return np.cumsum(bridge["intercept"]+bridge["beta"]*slopes)
