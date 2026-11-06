"""Analytical invariants, independent reference calculations and leakage tests."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from scipy import signal
from tslab.smoothing import CausalSmoother, FilterSpec, smooth
from tslab.data import validate_series, load_snapshot, digest
from tslab.targets import make_targets, mature_labels
from tslab.evaluation import null_predictions, benchmark, scorecard, paired_block_difference
from tslab.diagnostics import audit_replay, response_metrics

KINDS = ["identity", "sma", "wma", "ema", "dema", "tema", "kama", "sg",
         "local_linear", "spline", "butter", "bessel", "kalman"]


@pytest.mark.parametrize("kind", KINDS)
def test_prefix_stream_checkpoint_poison(kind):
    x = np.random.default_rng(42).normal(size=180).cumsum()*.01+4
    assert all(v for k, v in audit_replay(x, FilterSpec(kind=kind)).items() if k != "kind")


@pytest.mark.parametrize("kind", KINDS)
def test_constant(kind):
    out = smooth(np.full(130, 7.), FilterSpec(kind=kind))
    assert np.isnan(out[:59]).all()
    np.testing.assert_allclose(out[59:], 7., atol=1e-9)


def test_ema_against_pandas_and_analytic_forecast():
    x = np.linspace(4, 5, 130)**2
    spec = FilterSpec(warmup=1)
    engine = CausalSmoother(spec)
    s = engine.update_many(x)
    np.testing.assert_allclose(s, pd.Series(x).ewm(alpha=.2, adjust=False).mean())
    expected = .8**np.arange(1, 6)*s[-1] + (1-.8**np.arange(1, 6))*x[-1]
    np.testing.assert_allclose(engine.project(np.repeat(x[-1], 5)), expected)
    metrics, _ = response_metrics(spec)
    assert metrics["white_noise_gain"] == pytest.approx(.2/1.8)
    assert metrics["low_frequency_delay"] == pytest.approx(4.)


@pytest.mark.parametrize("kind", ["dema", "tema", "sg", "local_linear", "spline"])
def test_ramp_reproduction(kind):
    x = np.arange(600)*.01 + 4
    result = smooth(x, FilterSpec(kind=kind))
    np.testing.assert_allclose(result[-30:], x[-30:], atol=1e-8)


@pytest.mark.parametrize("kind", ["butter", "bessel"])
def test_iir_reference(kind):
    x = np.random.default_rng(12).normal(size=200)+4
    spec = FilterSpec(kind=kind, warmup=1)
    fn = signal.butter if kind == "butter" else signal.bessel
    kwargs = {"norm": "mag"} if kind == "bessel" else {}
    sos = fn(2, .05, fs=1, output="sos", **kwargs)
    ref, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos)*x[0])
    np.testing.assert_allclose(smooth(x, spec), ref)


def test_sg_independent_least_squares():
    x = np.random.default_rng(9).normal(size=100)
    spec = FilterSpec(kind="sg", window=11, degree=2, warmup=1)
    expected = np.polyval(np.polyfit(np.arange(11), x[-11:], 2), 10)
    assert smooth(x, spec)[-1] == pytest.approx(expected)


def test_kalman_covariance_and_resume():
    engine = CausalSmoother(FilterSpec(kind="kalman"))
    for x in np.random.default_rng(2).normal(size=150):
        engine.update(x)
        cov = np.array(engine.state["cov"])
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() >= -1e-14
    assert CausalSmoother.restore(json.loads(json.dumps(engine.checkpoint()))).checkpoint() == engine.checkpoint()


def sample():
    dates = pd.DatetimeIndex(["2026-12-23", "2026-12-24", "2026-12-28",
                              "2026-12-29", "2026-12-30", "2026-12-31", "2027-01-04"])
    x = pd.Series(np.arange(7, dtype=float), index=dates)
    return x, x*.5


def test_targets_holidays_delay_and_maturity():
    x, s = sample()
    targets = make_targets(x, s)
    row = targets.query("track == 'returns' and origin_pos == 0 and horizon == 5").iloc[0]
    assert row.target_end == x.index[5] and row.actual == 5
    delayed = make_targets(x, s, entry_delay=1)
    row = delayed.query("track == 'returns' and origin_pos == 0 and horizon == 5").iloc[0]
    assert row.target_start == x.index[1] and row.target_end == x.index[6] and row.actual == 5
    mature = mature_labels(targets, x.index[3])
    assert (mature.available_at <= x.index[3]).all()
    complete = mature_labels(targets, x.index[5], require_all_horizons=True)
    assert complete.groupby(["track", "origin"]).size().eq(5).all()
    assert complete.origin.nunique() == 1
    # Unavailable outcome values cannot change label eligibility.
    poisoned = targets.copy()
    poisoned.loc[poisoned.available_at > x.index[3], "actual"] = 99999
    pd.testing.assert_frame_equal(mature_labels(poisoned, x.index[3]), mature)


def test_target_units_and_warmup():
    x, s = sample()
    s.iloc[:2] = np.nan
    table = make_targets(x, s)
    assert table.query("track=='smoothed' and origin_pos==0 and horizon==1").actual.isna().all()
    assert table.query("origin_pos==6").actual.isna().all()


@pytest.mark.parametrize("mode", ["duplicate", "unsorted", "nan", "interior_nan"])
def test_invalid_data(mode):
    x, _ = sample()
    if mode == "duplicate":
        x.index = pd.DatetimeIndex([x.index[0]]*len(x))
    elif mode == "unsorted":
        x = x.iloc[::-1]
    elif mode == "nan":
        x.iloc[0] = np.nan
    else:
        x.iloc[3] = np.nan
    with pytest.raises(ValueError):
        validate_series(x, allow_warmup=mode == "interior_nan")


@pytest.mark.parametrize("kwargs", [{"kind": "bad"}, {"alpha": 0}, {"cutoff": .5},
    {"window": 2, "degree": 2}, {"r": 0}, {"lam": -1}, {"warmup": 1.2}, {"q_slope": np.nan}])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError):
        FilterSpec(**kwargs)


def test_snapshot_validation(tmp_path):
    path = tmp_path/"prices.csv"
    path.write_text("Date,Close\n2026-01-02,100\n2026-01-05,101\n")
    meta = tmp_path/"metadata.json"
    metadata = dict(sha256=digest(path), ticker="TEST", source="test", synthetic=True,
                    adjustment="none", calendar="explicit_test_sessions")
    meta.write_text(json.dumps(metadata))
    x, _ = load_snapshot(path, meta)
    assert x.iloc[1]-x.iloc[0] == pytest.approx(np.log(1.01))
    path.write_text(path.read_text().replace("101", "102"))
    with pytest.raises(ValueError, match="checksum"):
        load_snapshot(path, meta)


def test_forecast_poison_scores_and_latest():
    dates = pd.bdate_range("2020-01-01", periods=150)
    x = pd.Series(4+np.random.default_rng(8).normal(0, .01, len(dates)).cumsum(), index=dates)
    spec = FilterSpec()
    records, _ = benchmark(x, spec, start=70, step=5)
    cut = 110
    poisoned = x.copy()
    poisoned.iloc[cut:] += 50
    other, _ = benchmark(poisoned, spec, start=70, step=5)
    fields = ["origin", "track", "horizon", "model", "prediction"]
    pd.testing.assert_frame_equal(records.loc[records.origin < dates[cut], fields].reset_index(drop=True),
                                  other.loc[other.origin < dates[cut], fields].reset_index(drop=True))
    assert records.loc[records.origin == dates[-1], "status"].eq("immature").all()
    card = scorecard(records)
    e = records.query("track=='returns' and model=='zero' and horizon==1 and status=='scored'").error
    row = card.query("track=='returns' and model=='zero' and horizon==1").iloc[0]
    assert row.RMSE == pytest.approx(np.sqrt(np.mean(e.to_numpy()**2)))
    assert row.MSE_skill == 0
    assert card.groupby(["filter_id", "track", "horizon"]).n.nunique().eq(1).all()
    diff = paired_block_difference(records, "returns", "smooth_slope", "zero", block=3, draws=100)
    assert diff["lower95"] <= diff["upper95"]


def test_null_predictions_receive_only_history_and_preserve_state():
    x = np.arange(100)*.01
    engine = CausalSmoother()
    s = engine.update_many(x)
    state = engine.checkpoint()
    pred = null_predictions(x, s, engine)
    assert engine.checkpoint() == state
    assert len(pred) == 35
    with pytest.raises(ValueError):
        engine.update(np.nan)


def test_spline_leading_warmup_and_nonrevision():
    spec = FilterSpec(kind="spline", window=21, warmup=1)
    x = np.random.default_rng(3).normal(size=60)
    s = smooth(x, spec)
    assert np.isnan(s[:20]).all() and np.isfinite(s[20:]).all()
    np.testing.assert_allclose(s[:40], smooth(x[:40], spec), equal_nan=True)
    from scipy.interpolate import make_smoothing_spline
    direct = make_smoothing_spline(np.arange(21, dtype=float), x[-21:], lam=spec.lam)(20)
    assert s[-1] == pytest.approx(direct, abs=1e-11)
