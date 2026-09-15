# AGENTS.md

## Project

Explainable AI for Industrial Anomaly Root-Cause Analysis
and Counterfactual Decision Support.

## Fixed Research Direction

MetroPT industrial sensor data
→ preprocessing
→ time-aware windowing
→ anomaly detection
→ SHAP + LIME
→ explanation reliability
→ potential root-cause ranking
→ constrained counterfactual analysis
→ decision-support output

Do not change this research direction without explicit approval.

## Current Milestone

Month 1.

Month 1 objectives:

1. MetroPT dataset ingestion
2. Dataset schema and data dictionary
3. Exploratory data analysis
4. Failure/anomaly timeline
5. Leakage audit
6. Chronological train/validation/test split
7. Isolation Forest baseline
8. One-Class SVM baseline
9. Autoencoder baseline if computationally feasible
10. SHAP explanations
11. LIME explanations
12. SHAP/LIME agreement
13. Perturbation stability
14. Initial temporal consistency
15. Experiment logging
16. Reproducibility

## Scientific Rules

- Never use random train/test splitting.
- Prevent temporal leakage.
- Prevent preprocessing leakage.
- Prevent threshold leakage.
- Prevent window/event leakage.
- Fit preprocessing only on training data.
- Do not select thresholds using the test set.
- Do not describe feature attribution as physical causation.
- Never fabricate experimental results.
- Never fabricate literature gaps.
- Record experiment configuration.
- Record random seeds.
- Record git commit.
- Record metrics.
- Record runtime.
- Prefer event-level evaluation where appropriate.
- Treat temporal observations as dependent observations.
- Clearly distinguish model attribution from physical causality.

## Dataset Rules

- Use the public MetroPT dataset.
- Never commit the dataset to Git.
- Never commit large generated datasets to Git.
- Do not repeatedly load the raw CSV.
- Prefer Parquet/Polars/DuckDB for large-data processing.
- Preserve raw data separately from processed data.
- Record dataset provenance and version/checksum where practical.

## Engineering Rules

- Python 3.11.
- Keep CPU execution possible.
- GPU acceleration may be optional.
- Use modular functions.
- Write tests for new functionality.
- Avoid unnecessary dependencies.
- Keep configuration separate from source code.
- Use deterministic random seeds where applicable.
- Do not commit API keys or secrets.

## Experiment Rules

Every experiment should record, where applicable:

- experiment ID
- model
- parameters
- random seed
- dataset version
- data split
- preprocessing configuration
- threshold
- metrics
- runtime
- hardware
- git commit

## Codex Workflow

Before significant changes:

1. Inspect the relevant files.
2. Explain the proposed implementation.
3. Identify scientific risks.
4. Implement only the requested scope.
5. Add/update tests.
6. Run the tests.
7. Inspect the diff.
8. Report what changed.
9. Report remaining risks.

Do not silently change research methodology.

Do not implement future-month features prematurely.