"""Small, reproducible end-to-end anomaly analysis for the local prototype.

Feature rankings are model-agnostic deviation evidence, not causal explanations.
The detector and scaler are fitted on the chronological training partition only.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from xai_industrial.xai.agreement import jaccard


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    timeline: pd.DataFrame
    explanations: pd.DataFrame
    selected_row: dict[str, Any]


def infer_timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if str(col).strip().lower() in {"timestamp", "time", "datetime", "date"}:
            return str(col)
    return None


def _robust_deviation(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    """Absolute robust z-score relative to training data, with safe scale fallbacks."""
    center = np.nanmedian(train_values, axis=0)
    mad = np.nanmedian(np.abs(train_values - center), axis=0) * 1.4826
    q75, q25 = np.nanpercentile(train_values, [75, 25], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.nanstd(train_values, axis=0)
    scale = np.where(mad > 1e-9, mad, np.where(iqr_scale > 1e-9, iqr_scale, std_scale))
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    return np.abs((values - center) / scale)


def analyze_dataframe(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    contamination: float = 0.05,
    seed: int = 42,
    max_rows: int = 100_000,
    perturbation_repeats: int = 20,
    perturbation_scale: float = 0.01,
) -> AnalysisResult:
    """Analyze an uploaded table without fitting on its validation/test tail.

    Input should already be bounded by the caller. The first 70% of rows are used
    for fitting; the remainder are scored only. A time column is parsed/sorted
    when present. This is a prototype split, not event-purged research evaluation.
    """
    if not 0.001 <= float(contamination) <= 0.25:
        raise ValueError("contamination must be between 0.001 and 0.25")
    if frame.empty:
        raise ValueError("The uploaded dataset is empty.")
    if len(frame) > max_rows:
        raise ValueError(f"Dataset has {len(frame):,} rows; the prototype limit is {max_rows:,}.")

    df = frame.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError("Column names must be unique after trimming whitespace.")

    ts_col = timestamp_column or infer_timestamp_column(df)
    if ts_col and ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df = df.loc[df[ts_col].notna()].sort_values(ts_col, kind="stable").reset_index(drop=True)
    else:
        ts_col = None
        df = df.reset_index(drop=True)
    if len(df) < 30:
        raise ValueError("At least 30 valid rows are required for a meaningful prototype analysis.")

    excluded = {ts_col} if ts_col else set()
    sensor_cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    # Also accept numeric values represented as strings, but reject mostly-invalid columns.
    for c in df.columns:
        if c in excluded or c in sensor_cols:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().mean() >= 0.9:
            df[c] = converted
            sensor_cols.append(c)
    sensor_cols = [c for c in sensor_cols if df[c].notna().sum() >= max(30, int(0.5 * len(df)))]
    if not sensor_cols:
        raise ValueError("No usable numeric sensor columns were found.")

    Xdf = df[sensor_cols].replace([np.inf, -np.inf], np.nan)
    # Learn imputation values from the training prefix only.
    split_at = max(20, int(len(df) * 0.70))
    if split_at >= len(df):
        split_at = len(df) - 1
    train_medians = Xdf.iloc[:split_at].median().fillna(0.0)
    Xdf = Xdf.fillna(train_medians).fillna(0.0)
    # Constant features provide no useful ranking signal and can destabilize explanations.
    nonconstant = [c for c in sensor_cols if Xdf[c].iloc[:split_at].nunique(dropna=False) > 1]
    if not nonconstant:
        raise ValueError("All usable sensor columns are constant in the training partition.")
    sensor_cols = nonconstant
    X = Xdf[sensor_cols].to_numpy(dtype=float)
    X_train = X[:split_at]

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)
    model = IsolationForest(
        n_estimators=150, contamination=float(contamination), random_state=int(seed), n_jobs=-1
    )
    model.fit(X_train_scaled)
    raw_scores = -model.decision_function(X_scaled)
    flags = model.predict(X_scaled) == -1

    # Higher score means more anomalous. Rank globally for inspection only; this is not a calibrated probability.
    score_percentile = pd.Series(raw_scores).rank(pct=True).to_numpy() * 100.0
    timeline = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "anomaly_score": raw_scores,
        "score_percentile": score_percentile,
        "is_anomaly": flags,
    })
    if ts_col:
        timeline.insert(1, "timestamp", df[ts_col].astype(str).to_numpy())

    # Rank each sensor by absolute deviation from the training distribution for the most anomalous flagged row.
    candidate_indices = np.flatnonzero(flags)
    if len(candidate_indices) == 0:
        candidate_indices = np.array([int(np.argmax(raw_scores))])
    selected_idx = int(candidate_indices[np.argmax(raw_scores[candidate_indices])])
    base_dev = _robust_deviation(X[selected_idx:selected_idx + 1], X_train)[0]
    base_order = np.argsort(-base_dev)
    explanations = pd.DataFrame({
        "sensor": np.asarray(sensor_cols)[base_order],
        "deviation_score": base_dev[base_order],
        "observed_value": X[selected_idx, base_order],
        "training_median": np.median(X_train, axis=0)[base_order],
    })
    explanations["rank"] = np.arange(1, len(explanations) + 1)
    explanations["interpretation"] = "Deviation from training behaviour; candidate contributor, not proven cause"

    # Perturb the selected observation slightly and compare top-k rankings.
    rng = np.random.default_rng(seed)
    feature_scale = np.std(X_train, axis=0)
    feature_scale = np.where(feature_scale > 1e-9, feature_scale, 1.0)
    rankings: list[set[str]] = []
    k = min(5, len(sensor_cols))
    for _ in range(max(2, int(perturbation_repeats))):
        perturbed = X[selected_idx] + rng.normal(0.0, float(perturbation_scale), size=len(sensor_cols)) * feature_scale
        dev = _robust_deviation(perturbed.reshape(1, -1), X_train)[0]
        rankings.append(set(np.asarray(sensor_cols)[np.argsort(-dev)[:k]].tolist()))
    pairwise = [jaccard(a, b) for a, b in combinations(rankings, 2)]
    stability = float(np.mean(pairwise)) if pairwise else 1.0

    summary = {
        "rows_analyzed": int(len(df)),
        "sensor_count": int(len(sensor_cols)),
        "training_rows": int(split_at),
        "scored_rows": int(len(df) - split_at),
        "anomaly_count": int(flags.sum()),
        "anomaly_rate_pct": float(flags.mean() * 100.0),
        "selected_row_index": selected_idx,
        "selected_anomaly_score": float(raw_scores[selected_idx]),
        "selected_score_percentile": float(score_percentile[selected_idx]),
        "top_k_jaccard_stability": stability,
        "stability_repeats": int(max(2, int(perturbation_repeats))),
        "timestamp_column": ts_col,
        "sensor_columns": sensor_cols,
        "model": "IsolationForest",
        "contamination_setting": float(contamination),
        "seed": int(seed),
        "interpretation_note": "Anomaly scores are not probabilities. Sensor rankings indicate model-independent deviation evidence, not physical causality.",
    }
    selected_row: dict[str, Any] = {
        "row_index": selected_idx,
        "anomaly_score": float(raw_scores[selected_idx]),
        "score_percentile": float(score_percentile[selected_idx]),
        "is_anomaly": bool(flags[selected_idx]),
    }
    if ts_col:
        selected_row["timestamp"] = str(df.loc[selected_idx, ts_col])
    for c in sensor_cols:
        selected_row[c] = float(X[selected_idx, sensor_cols.index(c)])
    return AnalysisResult(summary, timeline, explanations, selected_row)
