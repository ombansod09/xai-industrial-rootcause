
from pathlib import Path

import pandas as pd

from xai_industrial.pipeline import analyze_dataframe

dataset = Path("data/processed/metropt.parquet")
if not dataset.is_file():
    raise FileNotFoundError(f"Dataset not found: {dataset.resolve()}")

# Read a bounded, chronological sample for the smoke test.
frame = pd.read_parquet(dataset).head(100_000)

result = analyze_dataframe(
    frame,
    contamination=0.05,
    seed=42,
    max_rows=100_000,
    perturbation_repeats=20,
    perturbation_scale=0.01,
)

summary = result.summary
selected_index = summary["selected_row_index"]
training_rows = summary["training_rows"]
scored_rows = summary["scored_rows"]

assert training_rows + scored_rows == summary["rows_analyzed"]
assert training_rows <= selected_index < summary["rows_analyzed"]
assert len(result.timeline) == summary["rows_analyzed"]
assert 0.0 <= summary["top_k_jaccard_stability"] <= 1.0

print("Smoke test completed")
print(f"Rows analyzed: {summary['rows_analyzed']}")
print(f"Training rows: {training_rows}")
print(f"Scored rows: {scored_rows}")
print(f"Selected row: {selected_index}")
print(f"Selected timestamp: {result.selected_row.get('timestamp')}")
print(f"Selected flagged: {result.selected_row['is_anomaly']}")
print(f"Anomaly score: {summary['selected_anomaly_score']:.6f}")
print(f"Top-k Jaccard stability: {summary['top_k_jaccard_stability']:.4f}")
print(f"Stability repeats: {summary['stability_repeats']}")
print("Scored-tail selection: PASS")
