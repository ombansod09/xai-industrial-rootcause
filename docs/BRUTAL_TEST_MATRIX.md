# Brutal Validation Matrix

| Test | Threat |
|---|---|
| Temporal leakage | Future data enters training |
| Scaling leakage | Test data influences scaler |
| Threshold leakage | Test labels tune threshold |
| Window leakage | Same event crosses splits |
| Seed sensitivity | One seed creates the result |
| Noise robustness | Small noise breaks model |
| Sensor dropout | Missing block breaks model |
| Drift | Performance collapses under shift |
| Correlation stress | Attribution swaps between correlated sensors |
| Explanation stability | Small perturbation changes ranking |
| Faithfulness | Plausible explanation is not causally useful |
| Cross-method | SHAP/LIME disagree |
| Cross-model | Detectors disagree |
| Counterfactual validity | Impossible values proposed |
| Reproducibility | Result cannot be recreated |
| Compute | Runtime/memory is impractical |
