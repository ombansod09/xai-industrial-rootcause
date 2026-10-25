# XAI Industrial Anomaly Root-Cause Analysis

Local prototype for industrial sensor anomaly detection, transparent sensor-deviation evidence, explanation-ranking stability, and incident export.

## Research direction

Industrial sensor data -> preprocessing/time windows -> anomaly detection -> SHAP/LIME (planned) -> explanation reliability -> potential root-cause ranking -> constrained counterfactual analysis (planned) -> decision support.

Primary dataset: MetroPT predictive-maintenance benchmark. Official record: https://zenodo.org/records/6854240

**Do not commit raw or processed datasets to Git.**

## Prototype quickstart

Use Python 3.11 or 3.12 as declared in `pyproject.toml`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,dashboard]"
pytest -q
streamlit run app.py
```

The dashboard accepts a CSV upload (maximum 20 MB and 100,000 rows) or provides a clearly labelled synthetic demonstration dataset. Parquet ingestion is available through the data module and conversion script but is not currently enabled in the dashboard.

## What currently works

- CSV-to-Parquet ingestion helper using Polars.
- Chronological split helper and baseline Isolation Forest factory.
- End-to-end prototype analysis: timestamp sorting, training-prefix imputation/scaling/model fitting, anomaly scoring, robust sensor deviation ranking, and top-k perturbation stability.
- Streamlit dashboard with anomaly timeline, sensor evidence table, limitations, and JSON/CSV export.
- Unit tests for split ordering, explanation utilities, analysis outputs, input validation, and incident report serialization.

## Important scientific limitations

- Sensor deviation rankings are not SHAP/LIME explanations and do not establish physical causation.
- The current prototype uses a simple chronological 70/30 fit/score split, but does not yet implement event-purged boundaries or the configured `gap_seconds`.
- `contamination` is a model setting, not an empirically established failure rate.
- Stability is top-k Jaccard overlap under small synthetic perturbations, not calibrated confidence or proof of faithfulness.
- The prototype does not yet include LIME/SHAP comparison, constrained counterfactuals, production authentication, or industrial control integration.
- Do not use it to make automatic machine-control decisions.

## Existing research protocol

See `AGENTS.md`, `docs/RESEARCH_PROTOCOL.md`, `docs/BRUTAL_TEST_MATRIX.md`, and `docs/DAY_30_DELIVERABLE.md` before changing research methodology.
