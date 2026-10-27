import numpy as np
import pandas as pd
import pytest

from xai_industrial.pipeline import analyze_dataframe


def sample_frame(n=160):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "pressure": rng.normal(10, 0.3, n),
        "temperature": rng.normal(50, 1.0, n),
        "vibration": rng.normal(0.2, 0.02, n),
    })
    frame.loc[n - 3:n - 1, ["pressure", "vibration"]] += [3.0, 0.5]
    return frame


def test_pipeline_returns_expected_outputs_and_stability():
    result = analyze_dataframe(sample_frame(), perturbation_repeats=5)
    assert result.summary["rows_analyzed"] == 160
    assert result.summary["training_rows"] == 112
    assert result.selected_row["row_index"] >= result.summary["training_rows"]
    assert result.summary["sensor_count"] == 3
    assert 0.0 <= result.summary["top_k_jaccard_stability"] <= 1.0
    assert {"sensor", "deviation_score", "rank"}.issubset(result.explanations.columns)
    assert result.timeline["anomaly_score"].notna().all()


def test_pipeline_rejects_empty_data():
    with pytest.raises(ValueError, match="empty"):
        analyze_dataframe(pd.DataFrame())


def test_pipeline_rejects_too_few_rows():
    with pytest.raises(ValueError, match="30 valid rows"):
        analyze_dataframe(sample_frame(20))


def test_pipeline_rejects_invalid_contamination():
    with pytest.raises(ValueError, match="contamination"):
        analyze_dataframe(sample_frame(), contamination=0.8)


def test_pipeline_purge_gap_preserves_partition_boundaries():
    result = analyze_dataframe(sample_frame(), purge_rows=10)

    assert result.summary["rows_analyzed"] == 160
    assert result.summary["training_rows"] == 112
    assert result.summary["purged_rows"] == 10
    assert result.summary["scored_rows"] == 38

    assert result.timeline["is_training"].sum() == 112
    assert result.timeline["is_purged"].sum() == 10
    assert result.timeline["is_evaluation"].sum() == 38

    assert not result.timeline.loc[112:121, "is_evaluation"].any()
    assert result.timeline.loc[122:, "is_evaluation"].all()
    assert result.selected_row["row_index"] >= 122


def test_pipeline_zero_purge_preserves_scored_row_count():
    result = analyze_dataframe(sample_frame(), purge_rows=0)

    assert result.summary["training_rows"] == 112
    assert result.summary["purged_rows"] == 0
    assert result.summary["scored_rows"] == 48
    assert result.timeline["is_evaluation"].sum() == 48
    assert result.selected_row["row_index"] >= 112


def test_pipeline_rejects_purge_that_leaves_no_evaluation_rows():
    with pytest.raises(ValueError, match="purge_rows leaves no evaluation rows"):
        analyze_dataframe(sample_frame(), purge_rows=48)


@pytest.mark.parametrize("purge_rows", [-1, 1.5, True])
def test_pipeline_rejects_invalid_purge_rows(purge_rows):
    with pytest.raises(ValueError, match="purge_rows"):
        analyze_dataframe(sample_frame(), purge_rows=purge_rows)
