"""Audit saved notebooks, input bytes, scorecards and output provenance."""
from pathlib import Path
from hashlib import sha256
from html.parser import HTMLParser
import json
import sys
import zipfile
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from tslab.data import digest, load_snapshot

market, meta = load_snapshot(ROOT/"inputs"/"SPY_adjusted_demo.csv",
                             ROOT/"inputs"/"SPY_adjusted_demo.metadata.json")
results = []


class FigureCounter(HTMLParser):
    def __init__(self):
        super().__init__()
        self.count = 0

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "img" and values.get("src", "").startswith("data:image/png;base64,"):
            assert values.get("alt"), "Missing figure alt text"
            self.count += 1


for path in sorted((ROOT/"notebooks").glob("TS*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    counts=[c.execution_count for c in cells]
    assert all(isinstance(n,int) and n>0 for n in counts) and all(b>a for a,b in zip(counts,counts[1:]))
    assert all(o.output_type != "error" for c in cells for o in c.outputs)
    images = sum("image/png" in o.get("data", {}) for c in cells for o in c.outputs)
    folder = ROOT/"outputs"/path.stem
    assert len(list(folder.glob("*.png"))) == images
    assert "Observed results" in path.with_suffix(".html").read_text()
    figures = FigureCounter()
    figures.feed(path.with_suffix(".html").read_text())
    assert figures.count == images
    manifest = json.loads((folder/"run_manifest.json").read_text())
    expected_config = sha256(json.dumps(manifest["config"],sort_keys=True,allow_nan=False).encode()).hexdigest()
    assert manifest["config_sha256"] == expected_config
    for filename, checksum in manifest["outputs_sha256"].items():
        assert digest(folder/filename) == checksum
    for filename, checksum in manifest["source_code_sha256"].items():
        if digest(ROOT/filename) == checksum:
            continue
        # Accepted milestone 1 notebooks retain their original execution evidence.
        # Check historical source bytes in the original immutable delivery archive.
        generation=nb.metadata.get('milestone',1)
        if generation>=8:
            raise AssertionError("Current source digest mismatch: "+filename)
        historical=ROOT/"provenance"/f"milestone_{generation}_sources"/filename
        if historical.exists() and digest(historical)!=checksum:
            historical=ROOT/'provenance/source_blobs'/f'{checksum}.py'
        if historical.exists():
            assert digest(historical)==checksum
        else:
            original_archive=ROOT.parent/f"Time_Series_Forecasting_Milestone_{generation}.zip"
            with zipfile.ZipFile(original_archive) as bundle:
                original=bundle.read("forecasting_library/"+filename)
            assert sha256(original).hexdigest()==checksum
            historical.parent.mkdir(parents=True,exist_ok=True)
            historical.write_bytes(original)
    if path.stem != "TS00_Foundations_and_Targets" and not path.stem.startswith(("TS09","TS10","TS11")):
        forecast = pd.read_csv(folder/"historical_forecasts.csv",
            parse_dates=["origin","target_start","target_end","available_at", "training_cutoff", "data_available_through"])
        assert (forecast.training_cutoff <= forecast.origin).all()
        assert (forecast.data_available_through <= forecast.origin).all()
        mature = forecast[forecast.status == "scored"]
        assert (mature.target_end == mature.available_at).all()
        assert (mature.target_start == mature.origin).all()
        assert (mature.target_end > mature.origin).all()
        # Verify every scored return and observed-session endpoint independently.
        ret = mature[mature.track == "returns"]
        positions = market.index.get_indexer(ret.origin)
        ends = positions+ret.horizon.to_numpy(dtype=int)
        assert (market.index[ends] == pd.DatetimeIndex(ret.target_end)).all()
        np.testing.assert_allclose(ret.actual,market.to_numpy()[ends]-market.to_numpy()[positions],atol=1e-12)
        assert forecast[forecast.origin==market.index[-1]].status.eq("immature").all()
        for track in ("smoothed","returns"):
            scores = pd.read_csv(folder/f"{track}_scorecard.csv")
            assert scores.groupby(["filter_id","horizon"]).n.nunique().eq(1).all()
    if path.stem.startswith(("TS04","TS05","TS06","TS07")):
        selection=pd.read_csv(folder/"selections.csv",parse_dates=["selection_cutoff","outer_first_origin","inner_last_label"])
        assert (selection.inner_last_label<=selection.selection_cutoff).all()
        assert (selection.selection_cutoff<selection.outer_first_origin).all()
        inner=pd.read_csv(folder/"inner_forecasts.csv",parse_dates=["available_at"])
        cutoffs=selection.groupby('block').selection_cutoff.first()
        assert (inner.available_at<=inner.block.map(cutoffs)).all()
        audit=pd.read_csv(folder/"fit_audit.csv")
        if path.stem.startswith('TS04'):
            assert (audit.bridge_last_label_pos<=audit.origin_pos).all()
        else:
            assert (audit.max_training_label<=audit.origin_pos).all()
        # Check both operational fallback values and frozen per-block choices.
        failed=forecast[forecast.fit_status=='fallback']
        for track,baseline in [('smoothed','filter_nochange'),('returns','zero')]:
            reference=forecast[(forecast.track==track)&(forecast.model==baseline)].set_index(['origin','horizon']).prediction
            for row in failed[failed.track==track].itertuples():
                assert row.prediction==reference.loc[(row.origin,row.horizon)]
        choices=selection.set_index(['block','track']).chosen_model
        point_lookup=forecast.set_index(['origin','track','horizon','model']).prediction
        for row in forecast[forecast.model=='selected_pipeline'].itertuples():
            assert row.chosen_model==choices.loc[(row.block,row.track)]
            assert row.prediction==point_lookup.loc[(row.origin,row.track,row.horizon,row.chosen_model)]
    if path.stem.startswith(('TS05','TS06','TS07')):
        from tslab.supervised import SupervisedData,FeatureSpec,chronological_holdout
        dataset=SupervisedData(market,FeatureSpec(**manifest['config']['features']))
        smoothed_rows=mature[mature.track=='smoothed']
        positions=market.index.get_indexer(smoothed_rows.origin)
        ends=positions+smoothed_rows.horizon.to_numpy(dtype=int)
        np.testing.assert_allclose(smoothed_rows.actual,dataset.smoothed.to_numpy()[ends],atol=1e-12)
        training_cache={}
        for row in audit[audit.fit_status=='ok'].itertuples():
            if str(row.seed)=='ensemble':continue
            key=(row.origin_pos,row.track)
            if key not in training_cache:
                training_cache[key]=dataset.training(row.origin_pos,row.track,manifest['config']['max_samples'])
            X,y,origin_positions=training_cache[key]
            assert row.max_training_label==origin_positions[-1]+5
            assert row.n_train==len(X)
            saved_mean=json.loads(row.feature_mean)
            if path.stem.startswith('TS05'):
                np.testing.assert_allclose(saved_mean,X.reshape(len(X),-1).mean(axis=0),atol=1e-14)
                target_mean=json.loads(row.target_mean)
                expected=y if len(target_mean)==5 else y[:,:1]
                np.testing.assert_allclose(target_mean,expected.mean(axis=0),atol=1e-14)
            else:
                tr,va=chronological_holdout(origin_positions,.2)
                assert row.optimization_last_label==origin_positions[tr[-1]]+5
                assert row.validation_first_origin==origin_positions[va[0]]
                assert row.optimization_last_label<row.validation_first_origin
                np.testing.assert_allclose(saved_mean,X.mean(axis=(0,1)),atol=1e-14)
                np.testing.assert_allclose(json.loads(row.stopping_feature_mean),X[tr].mean(axis=(0,1)),atol=1e-14)
        for filename,checksum in manifest.get('checkpoint_sha256',{}).items():
            assert digest(folder/filename)==checksum
        # Frozen comparison inputs must match the exact delivered reference bytes.
        for label,metadata in manifest['source']['references'].items():
            assert digest(ROOT/'inputs'/'learning_references'/f'{label}.csv')==metadata['sha256']
        if path.stem.startswith(('TS06','TS07')):
            import torch
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            from tslab.neural import NeuralFit
            members=pd.read_csv(folder/'seed_forecasts.csv',parse_dates=['origin'])
            member_means=members.groupby(['origin','track','horizon','model']).prediction.mean()
            for row in forecast[(forecast.model.isin([m['name'] for m in manifest['config']['models']])) & (forecast.fit_status=='ok')].itertuples():
                assert abs(row.prediction-member_means.loc[(row.origin,row.track,row.horizon,row.model)])<1e-12
            assert members.groupby(['origin','track','horizon','model']).seed.nunique().eq(len(manifest['config']['seeds'])).all()
            for checkpoint in sorted((folder/'checkpoints').glob('*.pt')):
                state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                fit=NeuralFit.restore(state)
                query=dataset.features[state['origin_pos']]
                np.testing.assert_array_equal(query,state['sequence'].numpy())
                predicted=fit.predict(query)
                np.testing.assert_array_equal(predicted,state['residual_prediction'].numpy())
                np.testing.assert_array_equal(dataset.decode(predicted,state['origin_pos'],state['track']),
                                               state['decoded_prediction'].numpy())
    if path.stem.startswith("TS08"):
        from verify_probability import verify_probability
        verify_probability(ROOT,folder,market,manifest)
    if path.stem.startswith("TS09"):
        from verify_cross_dataset import verify_cross_dataset
        verify_cross_dataset(ROOT,folder,manifest)
    if path.stem.startswith("TS10"):
        from verify_trading_integration import verify_trading_integration
        verify_trading_integration(ROOT,folder,manifest)
    if path.stem.startswith("TS11"):
        from verify_foundation import verify_foundation
        verify_foundation(ROOT,folder,manifest)
    results.append({"notebook":path.name,"code_cells":len(cells),"figures":images,
                    "csv_tables":len(list(folder.glob("*.csv"))),"status":"passed"})
summary = {"checks":"notebook structure, sequential outputs, images, hashes, actual labels and paired origins",
           "notebooks":results,"total_code_cells":sum(r["code_cells"] for r in results),
           "total_figures":sum(r["figures"] for r in results),
           "total_csv_tables":sum(r["csv_tables"] for r in results),
           "visual_check":"Separate direct inspection of exported figures; see VALIDATION.md"}
(ROOT/"artifact_verification.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
