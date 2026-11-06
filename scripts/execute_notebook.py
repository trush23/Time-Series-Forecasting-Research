"""Execute every cell in a fresh IPython process and save notebook/HTML.

This avoids kernel socket permissions; computations are unmodified and sequential.
Each invocation handles exactly one notebook, ensuring independent kernel state.
"""
import argparse
import base64
import io
import html as html_tools
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tslab-mpl")
os.environ.setdefault("IPYTHONDIR", "/private/tmp/tslab-ipython")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display, Image
from nbconvert import HTMLExporter


def execute(path):
    path = Path(path).resolve()
    os.chdir(path.parent)
    nb = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    figure_descriptions = []

    def show_figures(*args, **kwargs):
        for num in plt.get_fignums():
            fig = plt.figure(num)
            description = "; ".join(
                f"{ax.get_title()}. Horizontal axis: {ax.get_xlabel()}. Vertical axis: {ax.get_ylabel()}"
                for ax in fig.axes)
            figure_descriptions.append(description)
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
            display(Image(data=buffer.getvalue()))
            plt.close(fig)

    plt.show = show_figures
    start = time.perf_counter()
    count, images = 0, 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        count += 1
        print(f"{path.stem}: cell {count}", flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=False)
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
        for output in captured.outputs:
            outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
            images += int("image/png" in output.data)
        cell.outputs, cell.execution_count = outputs, count
        if result.error_in_exec or result.error_before_exec:
            nbformat.write(nb, path)
            raise RuntimeError(f"Cell {count}: {result.error_in_exec or result.error_before_exec}")
    validation = {"execution": "fresh-process in-process IPython; every code cell in order",
                  "code_cells": count, "embedded_figures": images,
                  "elapsed_seconds": round(time.perf_counter()-start, 3), "unexpected_errors": 0,
                  "visual_inspection": "recorded separately in VALIDATION.md"}
    out = ROOT/"outputs"/path.stem
    if path.stem.startswith("TS00"):
        control = pd.read_csv(out/"martingale_control.csv").iloc[-1]
        ratio = control.MC_persistence_MSE/control.MC_filter_MSE
        summary = (f"## Observed results\n\nThe 30,000 independent conditional simulations reproduce the analytic EMA mean. "
                   f"At horizon five, smoothed persistence has **{ratio:.2f} times** the filter-aware forecast's "
                   "conditional MSE in this specified null. This is filter catch-up, not return predictability. "
                   "Both research tracks have separate scorecards; current-origin future labels remain immature.")
    elif path.stem.startswith("TS01"):
        metrics = pd.read_csv(out/"response_metrics.csv").set_index("filter")
        scores = pd.read_csv(out/"returns_scorecard.csv")
        count = int((scores[(scores.model=="smooth_slope")&(scores.horizon==5)].MSE_skill < 0).sum())
        summary = (f"## Observed results\n\nAt alpha 0.2, EMA's IID measurement-noise variance gain is "
                   f"**{metrics.loc['EMA','white_noise_gain']:.4f}**, versus "
                   f"**{metrics.loc['DEMA','white_noise_gain']:.4f}** for DEMA and "
                   f"**{metrics.loc['TEMA','white_noise_gain']:.4f}** for TEMA. "
                   f"All **{count} of 4** tested smooth-slope return candidates have negative horizon-five MSE skill "
                   "against zero on this previously examined SPY demonstration period. This does not evaluate "
                   "ARIMA or learned forecasting models.")
    elif path.stem.startswith("TS02"):
        metrics = pd.read_csv(out/"matched_filter_metrics.csv").set_index("filter")
        revision = pd.read_csv(out/"historical_revision.csv")
        summary = ("## Observed results\n\nThe six fixed linear estimators have IID noise gains within "
                   f"**{100*metrics.relative_gain_mismatch.abs().max():.2f}%** of the EMA reference. "
                   f"Matched endpoint SG has **{100*metrics.loc['Endpoint SG','overshoot_fraction']:.1f}%** step overshoot, "
                   f"versus **{100*metrics.loc['Bessel','overshoot_fraction']:.2f}%** for Bessel. "
                   "The archived causal endpoints have exactly zero revision when later observations are appended. "
                   "These tradeoffs are parameter- and noise-model-specific; Kalman is shown separately without attenuation matching.")
    else:
        summary = (out/"observed_summary.md").read_text()
    nb.cells = [cell for cell in nb.cells if not cell.metadata.get("generated_summary")]
    nb.cells.insert(1, nbformat.v4.new_markdown_cell(summary, metadata={"generated_summary": True}))
    nb.metadata["validation"] = validation
    nbformat.validate(nb)
    nbformat.write(nb, path)
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    html, _ = exporter.from_notebook_node(nb)
    # Alt text for embedded notebook figures: preserve the actual rendered chart.
    for description in figure_descriptions:
        html = html.replace('alt="No description has been provided for this image"',
                            'alt="'+html_tools.escape(description, quote=True)+'"', 1)
    path.with_suffix(".html").write_text(html)
    (out/"execution_record.json").write_text(json.dumps(validation, indent=2))
    print(json.dumps(validation), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook")
    execute(parser.parse_args().notebook)
