from __future__ import annotations
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from xai_industrial.pipeline import analyze_dataframe
from xai_industrial.reporting.export import incident_json

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000

st.set_page_config(page_title="XAI-RCA | Industrial Monitor", page_icon="⚙️", layout="wide")
st.title("⚙️ XAI-RCA | Industrial Anomaly Monitor")
st.caption("Local research prototype · anomaly detection, sensor deviation evidence, and explanation stability")

with st.sidebar:
    st.header("Analysis settings")
    contamination = st.slider("Expected anomaly fraction", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=100_000, value=42, step=1)
    repeats = st.slider("Stability perturbation repeats", min_value=5, max_value=50, value=20, step=5)
    st.caption("The contamination setting controls the detector's flagged fraction. It is not a measured failure rate.")

uploaded = st.file_uploader("Upload a sensor CSV (max 20 MB)", type=["csv"])
use_demo = st.checkbox("Use clearly labelled synthetic demo data", value=uploaded is None)


def make_demo() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    n = 1200
    t = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    pressure = 7.0 + rng.normal(0, 0.12, n)
    temperature = 62 + rng.normal(0, 1.4, n)
    vibration = 0.35 + rng.normal(0, 0.035, n)
    flow = 40 + rng.normal(0, 1.8, n)
    # Synthetic injected drift for UI demonstration only, not real failure labels.
    pressure[930:950] += np.linspace(0.8, 2.2, 20)
    vibration[930:950] += 0.25
    temperature[930:950] += 6
    return pd.DataFrame({"timestamp": t, "pressure": pressure, "temperature": temperature, "vibration": vibration, "flow": flow})

frame = None
source_label = ""
if uploaded is not None:
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File exceeds the 20 MB prototype upload limit.")
        st.stop()
    try:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()), nrows=MAX_ROWS + 1, low_memory=False)
        source_label = f"Uploaded file: {uploaded.name}"
        if len(frame) > MAX_ROWS:
            st.error(f"This prototype supports up to {MAX_ROWS:,} rows per upload. Please use a representative sample.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read the CSV. Check its format. Details: {type(exc).__name__}")
        st.stop()
elif use_demo:
    frame = make_demo()
    source_label = "Synthetic demo data (not real industrial telemetry)"

if frame is None:
    st.info("Upload a CSV or enable synthetic demo data to begin.")
    st.stop()

st.caption(f"Source: **{source_label}** · {len(frame):,} rows · {len(frame.columns)} columns")
with st.expander("Preview input data"):
    st.dataframe(frame.head(10), use_container_width=True)

try:
    with st.spinner("Fitting on the chronological training prefix and scoring observations..."):
        result = analyze_dataframe(frame, contamination=contamination, seed=int(seed), max_rows=MAX_ROWS, perturbation_repeats=int(repeats))
except Exception as exc:
    st.error(f"Analysis could not complete: {exc}")
    st.stop()

s = result.summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows analyzed", f"{s['rows_analyzed']:,}")
c2.metric("Flagged anomalies", f"{s['anomaly_count']:,}", f"{s['anomaly_rate_pct']:.1f}% of rows")
c3.metric("Sensor features", str(s['sensor_count']))
c4.metric("Top-5 ranking stability", f"{s['top_k_jaccard_stability']:.2f}", help="Mean pairwise Jaccard overlap of top-k sensor sets under small perturbations. This measures ranking stability, not correctness.")

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Anomaly timeline")
    timeline = result.timeline.copy()
    xcol = "timestamp" if "timestamp" in timeline.columns else "row_index"
    fig = px.line(timeline, x=xcol, y="anomaly_score", title="Anomaly score (higher = more unusual)")
    flagged = timeline[timeline["is_anomaly"]]
    if not flagged.empty:
        fig.add_scatter(x=flagged[xcol], y=flagged["anomaly_score"], mode="markers", name="Flagged", marker={"size": 7})
    fig.update_layout(legend_title_text="", margin={"l": 10, "r": 10, "t": 45, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Selected observation")
    st.write(f"**Row:** {s['selected_row_index']}")
    st.write(f"**Score percentile:** {s['selected_score_percentile']:.1f}")
    st.write(f"**Model:** {s['model']}")
    st.warning("A flagged observation is an investigation candidate, not a confirmed equipment failure.")

st.subheader("Ranked sensor deviation evidence")
st.caption("Ranking is based on robust deviation from the training prefix. It is a transparent baseline, not SHAP/LIME and not a causal diagnosis.")
exp = result.explanations.head(10).copy()
bar = px.bar(exp.sort_values("deviation_score"), x="deviation_score", y="sensor", orientation="h", title="Largest deviations from training behaviour")
bar.update_layout(margin={"l": 10, "r": 10, "t": 45, "b": 10}, yaxis_title="Sensor", xaxis_title="Absolute robust deviation score")
st.plotly_chart(bar, use_container_width=True)
st.dataframe(exp[["rank", "sensor", "observed_value", "training_median", "deviation_score", "interpretation"]], use_container_width=True, hide_index=True)

with st.expander("Method and limitations"):
    st.markdown("""
- The data is sorted by a recognised timestamp column when available.
- Missing sensor values are filled using medians learned from the chronological training prefix.
- The scaler and Isolation Forest are fitted on the first 70% of rows; the remaining rows are scored without refitting.
- The displayed ranking measures deviations from training behaviour. Correlated sensors can share attribution, and deviation does not establish physical cause.
- Stability is measured as top-k set overlap under small synthetic perturbations. It is not a calibrated confidence score or proof of faithfulness.
- This prototype does not yet implement event-purged splitting, SHAP/LIME cross-method agreement, constrained counterfactuals, production authentication, or industrial control integration.
""")

json_report = incident_json(s, result.selected_row, result.explanations)
st.download_button("Download incident report (JSON)", data=json_report, file_name="xai_rca_incident_report.json", mime="application/json")
st.download_button("Download anomaly timeline (CSV)", data=result.timeline.to_csv(index=False).encode("utf-8"), file_name="xai_rca_anomaly_timeline.csv", mime="text/csv")
st.caption("Prototype only. Validate findings against machine context and maintenance evidence before operational use.")
