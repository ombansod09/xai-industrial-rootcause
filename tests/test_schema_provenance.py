import json

import pandas as pd
import pytest

from xai_industrial.data.load import convert_to_parquet
from xai_industrial.data.provenance import build_dataset_provenance
from xai_industrial.data.schema import validate_metropt_schema


def test_schema_validation_reports_configurable_schema():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="s"),
            "sensor": [1.0, None, 3.0],
        }
    )
    report = validate_metropt_schema(
        df,
        timestamp_column="timestamp",
        required_columns=("sensor",),
        expected_dtypes={"sensor": "numeric"},
    )
    assert report.row_count == 3
    assert report.missing_values["sensor"] == 1
    assert report.schema_summary["columns"] == ["timestamp", "sensor"]


def test_schema_validation_rejects_duplicate_timestamps():
    df = pd.DataFrame(
        {
            "timestamp": ["2026-01-01", "2026-01-01"],
            "sensor": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_metropt_schema(df, timestamp_column="timestamp")


def test_provenance_records_file_without_modifying_source(tmp_path):
    source = tmp_path / "sample.csv"
    source.write_text("timestamp,value\n2026-01-01,1\n", encoding="utf-8")
    before = source.read_bytes()
    provenance = build_dataset_provenance(
        source,
        source_identifier="local-test",
        row_count=1,
        schema_summary={"columns": ["timestamp", "value"]},
    )
    assert source.read_bytes() == before
    assert provenance.source_identifier == "local-test"
    assert provenance.source_filename == "sample.csv"
    assert len(provenance.sha256) == 64
    assert provenance.row_count == 1


def test_convert_to_parquet_writes_metadata_sidecar(tmp_path):
    source = tmp_path / "sample.csv"
    parquet = tmp_path / "sample.parquet"
    metadata = tmp_path / "sample.metadata.json"
    source.write_text(
        "timestamp,value\n2026-01-01T00:00:00,1\n2026-01-01T00:00:01,2\n",
        encoding="utf-8",
    )
    convert_to_parquet(
        source,
        parquet,
        metadata_path=metadata,
        source_identifier="unit-test-source",
    )
    assert parquet.exists()
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["source_identifier"] == "unit-test-source"
    assert payload["source_filename"] == "sample.csv"
    assert payload["row_count"] == 2
    assert payload["schema_summary"]["columns"] == ["timestamp", "value"]
