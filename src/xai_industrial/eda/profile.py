from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from xai_industrial.eda.failure_timeline import (
    empty_failure_timeline,
    failure_timeline_status,
    write_failure_timeline_csv,
)

UNKNOWN_DESCRIPTION = "Meaning not yet verified"


@dataclass(frozen=True)
class EdaArtifacts:
    summary_json: Path
    data_dictionary_csv: Path
    timestamp_profile_json: Path
    failure_timeline_csv: Path
    readme: Path


def run_eda(
    input_path: str | Path,
    *,
    metadata_path: str | Path | None,
    output_dir: str | Path,
    timestamp_column: str,
    config: dict[str, Any] | None = None,
) -> EdaArtifacts:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    profile = profile_parquet(
        input_path,
        metadata_path=metadata_path,
        timestamp_column=timestamp_column,
        config=config,
    )
    write_eda_outputs(profile, output_dir)
    return EdaArtifacts(
        summary_json=output_dir / "summary.json",
        data_dictionary_csv=output_dir / "data_dictionary.csv",
        timestamp_profile_json=output_dir / "timestamp_profile.json",
        failure_timeline_csv=output_dir / "failure_timeline.csv",
        readme=output_dir / "README.md",
    )


def profile_parquet(
    input_path: str | Path,
    *,
    metadata_path: str | Path | None,
    timestamp_column: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    lf = pl.scan_parquet(input_path)
    schema = lf.collect_schema()
    columns = list(schema.names())
    if timestamp_column not in columns:
        raise ValueError(f"Timestamp column {timestamp_column!r} is missing.")

    row_count = int(lf.select(pl.len()).collect().item())
    if row_count == 0:
        raise ValueError("EDA requires a non-empty dataset.")

    column_count = len(columns)
    dtypes = {name: str(dtype) for name, dtype in schema.items()}
    null_counts = _collect_single_row(lf.select(pl.all().null_count()))
    missing_values = {
        name: {
            "count": int(null_counts[name]),
            "percent": _percent(int(null_counts[name]), row_count),
        }
        for name in columns
    }
    unique_counts = _collect_single_row(
        lf.select([pl.col(name).n_unique().alias(name) for name in columns])
    )
    unique_counts = {name: int(unique_counts[name]) for name in columns}

    numeric_columns = [name for name, dtype in schema.items() if _is_numeric_dtype(dtype)]
    numeric_profile = _profile_numeric_columns(lf, numeric_columns)
    timestamp_profile = characterize_timestamps(lf, timestamp_column, row_count)
    duplicate_info = _profile_duplicates(lf, timestamp_column)
    categorical_profile = _profile_categorical_columns(
        lf, schema, unique_counts, row_count=row_count
    )
    data_dictionary = build_data_dictionary(
        columns=columns,
        dtypes=dtypes,
        missing_values=missing_values,
        unique_counts=unique_counts,
        numeric_profile=numeric_profile,
    )
    metadata = _load_metadata(metadata_path)
    metadata_warnings = _metadata_warnings(metadata, row_count, columns)
    failure_events = empty_failure_timeline()

    return {
        "summary": {
            "row_count": row_count,
            "column_count": column_count,
            "column_names": columns,
            "data_types": dtypes,
            "missing_values": missing_values,
            "duplicate_info": duplicate_info,
            "numeric_profile": numeric_profile,
            "categorical_profile": categorical_profile,
            "constant_columns": _constant_columns(unique_counts),
            "near_constant_columns": _near_constant_columns(unique_counts, row_count),
            "metadata_warnings": metadata_warnings,
            "reproducibility": _reproducibility(
                input_path=input_path,
                metadata_path=metadata_path,
                metadata=metadata,
                config=config or {},
            ),
            "failure_timeline_status": failure_timeline_status(failure_events),
        },
        "data_dictionary": data_dictionary,
        "timestamp_profile": timestamp_profile,
        "failure_events": failure_events,
    }


def characterize_timestamps(
    lf: pl.LazyFrame, timestamp_column: str, row_count: int
) -> dict[str, Any]:
    schema = lf.collect_schema()
    timestamp_expr = _timestamp_expr(timestamp_column, schema[timestamp_column])
    parsed = lf.select(timestamp_expr.alias("timestamp"))
    null_count = int(parsed.select(pl.col("timestamp").null_count()).collect().item())
    if null_count:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} has {null_count} null or unparsable values."
        )

    base = parsed.sort("timestamp").with_columns(
        pl.col("timestamp").diff().dt.total_seconds().alias("delta_seconds")
    )
    bounds = _collect_single_row(
        base.select(
            pl.col("timestamp").min().alias("first_timestamp"),
            pl.col("timestamp").max().alias("last_timestamp"),
            pl.len().alias("observation_count"),
            pl.col("timestamp").n_unique().alias("unique_timestamp_count"),
        )
    )
    duplicate_timestamp_count = row_count - int(bounds["unique_timestamp_count"])
    if duplicate_timestamp_count:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} has "
            f"{duplicate_timestamp_count} duplicate values."
        )

    delta_stats = _collect_single_row(
        base.select(
            pl.col("delta_seconds").drop_nulls().min().alias("min_seconds"),
            pl.col("delta_seconds").drop_nulls().max().alias("max_seconds"),
            pl.col("delta_seconds").drop_nulls().mean().alias("mean_seconds"),
            pl.col("delta_seconds").drop_nulls().median().alias("median_seconds"),
            pl.col("delta_seconds").drop_nulls().std().alias("std_seconds"),
            pl.col("delta_seconds").drop_nulls().n_unique().alias("unique_delta_count"),
        )
    )
    expected_interval = _expected_interval(delta_stats)
    large_gap_threshold = None
    large_gaps: list[dict[str, Any]] = []
    if expected_interval is not None and expected_interval > 0:
        large_gap_threshold = expected_interval * 3
        large_gaps = [
            _json_ready(row)
            for row in base.filter(pl.col("delta_seconds") > large_gap_threshold)
            .select("timestamp", "delta_seconds")
            .limit(100)
            .collect()
            .to_dicts()
        ]

    observed_deltas = (
        base.filter(pl.col("delta_seconds").is_not_null())
        .group_by("delta_seconds")
        .len()
        .sort(["len", "delta_seconds"], descending=[True, False])
        .limit(20)
        .collect()
        .to_dicts()
    )
    return {
        "timestamp_column": timestamp_column,
        "first_timestamp": _json_ready(bounds["first_timestamp"]),
        "last_timestamp": _json_ready(bounds["last_timestamp"]),
        "observation_count": int(bounds["observation_count"]),
        "expected_sampling_interval_seconds": expected_interval,
        "observed_time_differences": [_json_ready(row) for row in observed_deltas],
        "irregular_sampling": {
            "min_seconds": _json_ready(delta_stats["min_seconds"]),
            "max_seconds": _json_ready(delta_stats["max_seconds"]),
            "mean_seconds": _json_ready(delta_stats["mean_seconds"]),
            "median_seconds": _json_ready(delta_stats["median_seconds"]),
            "std_seconds": _json_ready(delta_stats["std_seconds"]),
            "unique_delta_count": int(delta_stats["unique_delta_count"] or 0),
        },
        "large_gap_threshold_seconds": large_gap_threshold,
        "large_gaps": large_gaps,
    }


def build_data_dictionary(
    *,
    columns: list[str],
    dtypes: dict[str, str],
    missing_values: dict[str, dict[str, float | int]],
    unique_counts: dict[str, int],
    numeric_profile: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        stats = numeric_profile.get(column, {})
        rows.append(
            {
                "column_name": column,
                "dtype": dtypes[column],
                "missing_count": missing_values[column]["count"],
                "missing_percent": missing_values[column]["percent"],
                "unique_count": unique_counts[column],
                "observed_min": stats.get("min"),
                "observed_max": stats.get("max"),
                "description": UNKNOWN_DESCRIPTION,
            }
        )
    return rows


def write_eda_outputs(profile: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "summary.json", profile["summary"])
    _write_json(output_dir / "timestamp_profile.json", profile["timestamp_profile"])
    _write_data_dictionary_csv(
        profile["data_dictionary"], output_dir / "data_dictionary.csv"
    )
    write_failure_timeline_csv(
        profile["failure_events"], output_dir / "failure_timeline.csv"
    )
    _write_readme(output_dir / "README.md", profile["summary"])


def _profile_numeric_columns(
    lf: pl.LazyFrame, numeric_columns: list[str]
) -> dict[str, dict[str, Any]]:
    if not numeric_columns:
        return {}
    exprs = []
    for column in numeric_columns:
        exprs.extend(
            [
                pl.col(column).min().alias(f"{column}__min"),
                pl.col(column).max().alias(f"{column}__max"),
                pl.col(column).mean().alias(f"{column}__mean"),
                pl.col(column).median().alias(f"{column}__median"),
                pl.col(column).std().alias(f"{column}__std"),
                pl.col(column).quantile(0.01).alias(f"{column}__q01"),
                pl.col(column).quantile(0.05).alias(f"{column}__q05"),
                pl.col(column).quantile(0.25).alias(f"{column}__q25"),
                pl.col(column).quantile(0.75).alias(f"{column}__q75"),
                pl.col(column).quantile(0.95).alias(f"{column}__q95"),
                pl.col(column).quantile(0.99).alias(f"{column}__q99"),
            ]
        )
    row = _collect_single_row(lf.select(exprs))
    out = {}
    for column in numeric_columns:
        out[column] = {
            "min": _json_ready(row[f"{column}__min"]),
            "max": _json_ready(row[f"{column}__max"]),
            "mean": _json_ready(row[f"{column}__mean"]),
            "median": _json_ready(row[f"{column}__median"]),
            "std": _json_ready(row[f"{column}__std"]),
            "quantiles": {
                "0.01": _json_ready(row[f"{column}__q01"]),
                "0.05": _json_ready(row[f"{column}__q05"]),
                "0.25": _json_ready(row[f"{column}__q25"]),
                "0.75": _json_ready(row[f"{column}__q75"]),
                "0.95": _json_ready(row[f"{column}__q95"]),
                "0.99": _json_ready(row[f"{column}__q99"]),
            },
        }
    return out


def _profile_duplicates(lf: pl.LazyFrame, timestamp_column: str) -> dict[str, int]:
    row = _collect_single_row(
        lf.select(
            pl.struct(pl.all()).is_duplicated().sum().alias("duplicate_row_count"),
            pl.col(timestamp_column).is_duplicated().sum().alias("duplicate_timestamp_count"),
        )
    )
    return {key: int(value) for key, value in row.items()}


def _profile_categorical_columns(
    lf: pl.LazyFrame,
    schema: pl.Schema,
    unique_counts: dict[str, int],
    *,
    row_count: int,
    max_unique: int = 20,
) -> dict[str, Any]:
    out = {}
    for column, dtype in schema.items():
        if _is_categorical_like(dtype) or unique_counts[column] <= max_unique:
            values = (
                lf.group_by(column)
                .len()
                .sort("len", descending=True)
                .limit(max_unique)
                .collect()
                .to_dicts()
            )
            out[column] = {
                "unique_count": unique_counts[column],
                "top_values": [_json_ready(row) for row in values],
                "is_low_cardinality": unique_counts[column] <= max_unique,
                "unique_fraction": _percent(unique_counts[column], row_count),
            }
    return out


def _timestamp_expr(column: str, dtype: pl.DataType) -> pl.Expr:
    expr = pl.col(column)
    if dtype == pl.String:
        return expr.str.strptime(pl.Datetime, strict=False)
    return expr.cast(pl.Datetime, strict=False)


def _expected_interval(delta_stats: dict[str, Any]) -> float | None:
    median = delta_stats.get("median_seconds")
    unique_count = int(delta_stats.get("unique_delta_count") or 0)
    if median is None or unique_count == 0:
        return None
    return float(median)


def _metadata_warnings(
    metadata: dict[str, Any] | None, row_count: int, columns: list[str]
) -> list[str]:
    if metadata is None:
        return []
    warnings = []
    metadata_row_count = metadata.get("row_count")
    if metadata_row_count is not None and int(metadata_row_count) != row_count:
        warnings.append(
            f"Metadata row_count={metadata_row_count} does not match Parquet row_count={row_count}."
        )
    metadata_columns = metadata.get("schema_summary", {}).get("columns")
    if metadata_columns is not None and list(metadata_columns) != columns:
        warnings.append("Metadata schema_summary.columns does not match Parquet columns.")
    return warnings


def _reproducibility(
    *,
    input_path: Path,
    metadata_path: str | Path | None,
    metadata: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "execution_timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "input_path": str(input_path),
        "input_name": input_path.name,
        "input_checksum_sha256": metadata.get("sha256") if metadata else None,
        "metadata_path": str(metadata_path) if metadata_path else None,
        "configuration": config,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "polars": pl.__version__,
        },
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_metadata(metadata_path: str | Path | None) -> dict[str, Any] | None:
    if metadata_path is None:
        return None
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Metadata sidecar must contain a JSON object.")
    return loaded


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def _write_data_dictionary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "column_name",
        "dtype",
        "missing_count",
        "missing_percent",
        "unique_count",
        "observed_min",
        "observed_max",
        "description",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# EDA Outputs",
            "",
            "Generated by the reproducible Day-2 EDA pipeline.",
            "",
            "Files:",
            "- summary.json",
            "- data_dictionary.csv",
            "- timestamp_profile.json",
            "- failure_timeline.csv",
            "",
            f"Rows profiled: {summary['row_count']}",
            f"Columns profiled: {summary['column_count']}",
            f"Failure timeline status: {summary['failure_timeline_status']}",
            "",
            "Column descriptions are marked as 'Meaning not yet verified' unless backed by verified documentation.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def _collect_single_row(lf: pl.LazyFrame) -> dict[str, Any]:
    rows = lf.collect().to_dicts()
    if not rows:
        return {}
    return rows[0]


def _constant_columns(unique_counts: dict[str, int]) -> list[str]:
    return [column for column, count in unique_counts.items() if count <= 1]


def _near_constant_columns(
    unique_counts: dict[str, int], row_count: int, threshold: float = 0.001
) -> list[str]:
    return [
        column
        for column, count in unique_counts.items()
        if count > 1 and (count / row_count) <= threshold
    ]


def _percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return float((numerator / denominator) * 100)


def _is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype.is_numeric()


def _is_categorical_like(dtype: pl.DataType) -> bool:
    return dtype in {pl.String, pl.Categorical, pl.Enum, pl.Boolean}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
