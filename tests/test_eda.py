import json

import polars as pl
import pytest

from xai_industrial.eda.failure_timeline import (
    FAILURE_TIMELINE_NOT_AVAILABLE,
    empty_failure_timeline,
    failure_timeline_status,
)
from xai_industrial.eda.profile import profile_parquet, run_eda


def _write_fixture(path):
    df = pl.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:10",
                "2026-01-01T00:00:20",
                "2026-01-01T00:01:20",
            ],
            "sensor_a": [1.0, 2.0, None, 4.0],
            "sensor_b": [5, 5, 5, 5],
            "state": ["on", "on", "off", "off"],
        }
    )
    df.write_parquet(path)


def test_eda_profiles_small_fixture(tmp_path):
    parquet = tmp_path / "fixture.parquet"
    _write_fixture(parquet)

    profile = profile_parquet(
        parquet,
        metadata_path=None,
        timestamp_column="timestamp",
        config={"seed": 42},
    )

    assert profile["summary"]["row_count"] == 4
    assert profile["summary"]["column_count"] == 4
    assert profile["summary"]["missing_values"]["sensor_a"]["count"] == 1
    assert profile["summary"]["constant_columns"] == ["sensor_b"]
    assert profile["summary"]["numeric_profile"]["sensor_a"]["min"] == 1.0
    assert profile["data_dictionary"][0]["description"] == "Meaning not yet verified"


def test_timestamp_characterization(tmp_path):
    parquet = tmp_path / "fixture.parquet"
    _write_fixture(parquet)

    profile = profile_parquet(
        parquet,
        metadata_path=None,
        timestamp_column="timestamp",
        config={},
    )
    timestamp_profile = profile["timestamp_profile"]

    assert timestamp_profile["first_timestamp"] == "2026-01-01T00:00:00"
    assert timestamp_profile["last_timestamp"] == "2026-01-01T00:01:20"
    assert timestamp_profile["observation_count"] == 4
    assert timestamp_profile["expected_sampling_interval_seconds"] == 10.0
    assert timestamp_profile["large_gaps"][0]["delta_seconds"] == 60


def test_run_eda_writes_deterministic_artifact_shapes(tmp_path):
    parquet = tmp_path / "fixture.parquet"
    output = tmp_path / "eda"
    _write_fixture(parquet)

    first = run_eda(
        parquet,
        metadata_path=None,
        output_dir=output,
        timestamp_column="timestamp",
        config={"seed": 42},
    )
    first_summary = json.loads(first.summary_json.read_text(encoding="utf-8"))
    second = run_eda(
        parquet,
        metadata_path=None,
        output_dir=output,
        timestamp_column="timestamp",
        config={"seed": 42},
    )
    second_summary = json.loads(second.summary_json.read_text(encoding="utf-8"))

    assert first_summary["row_count"] == second_summary["row_count"]
    assert first_summary["column_names"] == second_summary["column_names"]
    assert first.data_dictionary_csv.read_text(encoding="utf-8").startswith("column_name")
    assert second.failure_timeline_csv.exists()


def test_empty_dataset_fails_clearly(tmp_path):
    parquet = tmp_path / "empty.parquet"
    pl.DataFrame({"timestamp": [], "sensor": []}).write_parquet(parquet)

    with pytest.raises(ValueError, match="non-empty"):
        profile_parquet(parquet, metadata_path=None, timestamp_column="timestamp")


def test_missing_timestamp_fails(tmp_path):
    parquet = tmp_path / "missing_timestamp.parquet"
    pl.DataFrame({"sensor": [1, 2, 3]}).write_parquet(parquet)

    with pytest.raises(ValueError, match="Timestamp column"):
        profile_parquet(parquet, metadata_path=None, timestamp_column="timestamp")


def test_failure_timeline_no_verified_events():
    events = empty_failure_timeline()
    assert events == []
    assert failure_timeline_status(events) == FAILURE_TIMELINE_NOT_AVAILABLE


def test_metadata_sidecar_mismatch_warning(tmp_path):
    parquet = tmp_path / "fixture.parquet"
    metadata = tmp_path / "metadata.json"
    _write_fixture(parquet)
    metadata.write_text(
        json.dumps(
            {
                "row_count": 999,
                "sha256": "abc",
                "schema_summary": {"columns": ["wrong"]},
            }
        ),
        encoding="utf-8",
    )

    profile = profile_parquet(
        parquet,
        metadata_path=metadata,
        timestamp_column="timestamp",
        config={},
    )

    warnings = profile["summary"]["metadata_warnings"]
    assert any("row_count" in warning for warning in warnings)
    assert any("columns" in warning for warning in warnings)
