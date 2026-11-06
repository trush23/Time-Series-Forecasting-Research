"""Nested chronological selection with per-track choices and visible failures."""
import json
import time
import numpy as np
import pandas as pd
from .data import validate_series
from .smoothing import CausalSmoother, FilterSpec
from .targets import make_targets
from .evaluation import null_predictions, scorecard
from .classical import native_forecast, fit_return_bridge, bridge_forecast


class OriginRunner:
    """Cache origin-specific fits within ONE data snapshot, never across runs."""
    def __init__(self, log_price, models, filter_spec=FilterSpec(), train_window=252):
        validate_series(log_price)
        if not models or len({m.name for m in models}) != len(models):
            raise ValueError("Need nonempty, uniquely named candidate models")
        reserved = {"selected_pipeline","persistence","smooth_slope","filter_nochange","filter_drift","zero","raw_drift"}
        if {m.name for m in models}&reserved:
            raise ValueError("Candidate name collides with a benchmark")
        if train_window < 40:
            raise ValueError("train_window must be >=40")
        self.log_price, self.models = log_price, list(models)
        self.filter_spec, self.train_window = filter_spec, train_window
        self.cache, self.audit = {}, []
        # Build immutable causal endpoints/checkpoints; models receive prefixes.
        engine = CausalSmoother(filter_spec)
        self.states, endpoints = [], []
        for value in log_price:
            endpoints.append(engine.update(value))
            self.states.append(engine.checkpoint())
        self.smoothed = pd.Series(endpoints,index=log_price.index)
        self.targets = make_targets(log_price,self.smoothed)

    def predictions(self, origin_pos):
        t = origin_pos
        if isinstance(t,bool) or not isinstance(t,(int,np.integer)) or not 0 <= t < len(self.log_price):
            raise ValueError("Origin position is outside observed history")
        if t in self.cache:
            return self.cache[t].copy()
        x = self.log_price.iloc[:t+1].to_numpy()
        s = self.smoothed.iloc[:t+1].to_numpy()
        finite_s = s[np.isfinite(s)]
        engine = CausalSmoother.restore(self.states[t])
        base = null_predictions(x,s,engine)
        base["fit_status"] = "baseline"
        base["chosen_model"] = base.model
        outputs = [base]
        try:
            bridge = fit_return_bridge(x,s,self.train_window)
            bridge_error = ""
        except Exception as exc:
            bridge, bridge_error = None, f"{type(exc).__name__}: {exc}"
        lookup = base.set_index(["track","model","horizon"]).prediction
        for model in self.models:
            start = time.perf_counter()
            try:
                native, diagnostic = native_forecast(finite_s,model,train_window=self.train_window)
                native_error = ""
            except Exception as exc:
                native, diagnostic = None, {}
                native_error = f"{type(exc).__name__}: {exc}"
            for track in ("smoothed","returns"):
                error = native_error or (bridge_error if track=="returns" else "")
                status = "fallback" if error else "ok"
                baseline = "filter_nochange" if track=="smoothed" else "zero"
                point = (native if track=="smoothed" else bridge_forecast(native,s,bridge)) if not error else None
                for h in range(1,6):
                    outputs.append(pd.DataFrame([{"track":track,"horizon":h,"model":model.name,
                        "prediction":float(point[h-1]) if point is not None else lookup.loc[(track,baseline,h)],
                        "fit_status":status,"chosen_model":model.name}]))
                self.audit.append({"origin":self.log_price.index[t],"origin_pos":t,"track":track,
                    "model":model.name,"fit_status":status,"error":error,
                    "warnings":diagnostic.get("warnings",""),"n_train":diagnostic.get("n_train",0),
                    "runtime_seconds":time.perf_counter()-start,
                    "bridge_last_label_pos":bridge["last_label_pos"] if bridge else -1,
                    "bridge_n_pairs":bridge["n_pairs"] if bridge else 0,
                    "bridge_intercept":bridge["intercept"] if bridge else np.nan,
                    "bridge_beta":bridge["beta"] if bridge else np.nan})
        result = pd.concat(outputs,ignore_index=True)
        result["origin"], result["origin_pos"] = self.log_price.index[t], t
        result["training_cutoff"] = self.log_price.index[t]
        result["data_available_through"] = self.log_price.index[t]
        self.cache[t] = result
        return result.copy()

    def attach_labels(self, predictions):
        """Ground truth attaches after prediction; never handed to native models."""
        result = predictions.merge(self.targets.drop(columns="origin_pos"),
            on=["origin","track","horizon"],validate="many_to_one")
        result["filter_id"] = f"{self.filter_spec.kind}_alpha_{self.filter_spec.alpha}"
        result["units"] = np.where(result.track=="smoothed","log_level","log_return")
        result["error"] = result.prediction-result.actual
        result["status"] = np.where(result.actual.notna(),"scored","immature")
        return result


def choose_models(inner_records, candidate_names, min_success=.8):
    """Equal-horizon mean MAE / benchmark MAE; choices separate for each track.

    Require ALL candidate forecast rows to have matured before selection. Ties
    follow the declared candidate order. Poor operational availability is a
    predeclared exclusion, never retrospective removal of forecast origins.
    """
    if not 0 <= min_success <= 1:
        raise ValueError("Success threshold must be in [0,1]")
    if not inner_records.status.eq("scored").all():
        raise ValueError("Inner validation contains immature labels")
    scores = scorecard(inner_records)
    rows, choices = [], {}
    for track in ("smoothed","returns"):
        baseline = "filter_nochange" if track=="smoothed" else "zero"
        rates = inner_records[inner_records.track==track].groupby("model").fit_status.apply(lambda v:(v=="ok").mean())
        ranked = []
        for order,name in enumerate(candidate_names):
            part = scores[(scores.track==track)&(scores.model==name)]
            if len(part)!=5:
                raise ValueError("Every candidate must cover all five horizons")
            loss = float((1-part.MAE_skill).mean())
            success = float(rates[name])
            eligible = success >= min_success and np.isfinite(loss)
            rows.append({"track":track,"model":name,"relative_MAE":loss,
                         "successful_fit_fraction":success,"eligible":eligible})
            if eligible:
                ranked.append((loss,order,name))
        # Baseline is itself an eligible model, so selection never forces a bad fit.
        ranked.append((1.,len(candidate_names),baseline))
        choices[track] = min(ranked)[2]
    return choices, pd.DataFrame(rows)


def nested_benchmark(log_price, models, filter_spec=FilterSpec(), outer_starts=(1000,1400,1800),
                     outer_step=20, inner_count=6, inner_step=25, train_window=252,
                     min_success=.8):
    """Re-select before each outer block; fit parameters anew at every origin.

    Tuning includes the final five-session label at inner origins only when its
    endpoint precedes the outer block. Outer labels never enter that block's
    selection. Baselines can win. Latest origin is forecast but remains unscored.
    """
    starts = tuple(outer_starts)
    if (not starts or starts!=tuple(sorted(set(starts))) or
        any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) for v in starts) or
        any(not isinstance(v,int) or v<1 for v in [outer_step,inner_count,inner_step]) or
        starts[-1]>=len(log_price)):
        raise ValueError("Invalid outer/inner schedule")
    runner = OriginRunner(log_price,models,filter_spec,train_window)
    selection, tuning, inner_exports, outer_exports = [],[],[],[]
    names = [m.name for m in models]
    for block,start in enumerate(starts):
        inner = [start-6-inner_step*k for k in reversed(range(inner_count))]
        if inner[0] < filter_spec.minimum_history+40:
            raise ValueError("Insufficient initial history for inner validation")
        inner_predictions = pd.concat([runner.predictions(i) for i in inner],ignore_index=True)
        inner_records = runner.attach_labels(inner_predictions)
        cutoff = log_price.index[start-1]
        assert (inner_records.available_at <= cutoff).all()
        choices, table = choose_models(inner_records,names,min_success)
        table["block"] = block
        tuning.append(table)
        inner_records["block"] = block
        inner_exports.append(inner_records)
        stop = starts[block+1] if block+1<len(starts) else len(log_price)
        origins = list(range(start,stop,outer_step))
        if block==len(starts)-1 and origins[-1]!=len(log_price)-1:
            origins.append(len(log_price)-1)
        for track,chosen in choices.items():
            selection.append({"block":block,"track":track,"chosen_model":chosen,
                "selection_cutoff":cutoff,"outer_first_origin":log_price.index[start],
                "outer_last_origin":log_price.index[origins[-1]],
                "inner_first_origin":log_price.index[inner[0]],
                "inner_last_label":log_price.index[inner[-1]+5],
                "minimum_success_fraction":min_success,"objective":"equal-horizon relative MAE"})
        for t in origins:
            predictions = runner.predictions(t)
            selected = pd.concat([predictions[(predictions.track==track)&(predictions.model==chosen)]
                                  for track,chosen in choices.items()],ignore_index=True).copy()
            selected["chosen_model"] = selected.model
            selected["model"] = "selected_pipeline"
            predictions = pd.concat([predictions,selected],ignore_index=True)
            predictions["block"] = block
            predictions["selection_cutoff"] = cutoff
            outer_exports.append(runner.attach_labels(predictions))
    return {"forecasts":pd.concat(outer_exports,ignore_index=True),
            "inner_forecasts":pd.concat(inner_exports,ignore_index=True),
            "selections":pd.DataFrame(selection),"tuning":pd.concat(tuning,ignore_index=True),
            "fit_audit":pd.DataFrame(runner.audit),"smoothed":runner.smoothed}
