# XAI Industrial Anomaly Root-Cause Analysis

Fixed research pipeline:
Industrial sensor data -> preprocessing/time windows -> anomaly detection -> SHAP/LIME -> explanation reliability -> potential root-cause ranking -> constrained counterfactual analysis -> decision support.

Primary dataset: MetroPT predictive-maintenance benchmark.
Official record: https://zenodo.org/records/6854240

Do NOT commit the dataset to Git.

## Start
Python 3.11 or 3.12.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q
python scripts/smoke_test.py
```

## Day-30 target
Reproducible data pipeline, EDA, leakage audit, two anomaly baselines, first SHAP/LIME explanations, agreement/stability experiments, and initial research-gap evidence.
