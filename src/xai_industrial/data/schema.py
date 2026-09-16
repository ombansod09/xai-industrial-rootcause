from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)


@dataclass(frozen=True)
class SchemaValidationReport:
    row_count: int
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    timestamp_column: str
    duplicate_timestamp_count: int
    schema_summary: dict[str, Any]


def validate_metropt_schema(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    required_columns: tuple[str, ...] | list[str] = (),
    expected_dtypes: dict[str, str] | None = None,
) -> SchemaValidationReport:
    """Validate configurable MetroPT schema assumptions without inventing columns."""
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_column!r} is missing.")

    missing_required = [col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns are missing: {missing_required!r}.")

    expected_dtypes = expected_dtypes or {}
    for column, expected in expected_dtypes.items():
        if column not in df.columns:
            raise ValueError(f"Column {column!r} with expected dtype is missing.")
        if not _dtype_matches(df[column], expected):
            actual = str(df[column].dtype)
            raise ValueError(
                f"Column {column!r} has dtype {actual!r}, expected {expected!r}."
            )

    parsed_timestamp = pd.to_datetime(df[timestamp_column], errors="coerce")
    null_timestamp_count = int(parsed_timestamp.isna().sum())
    if null_timestamp_count:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} has {null_timestamp_count} null "
            "or unparsable values."
        )

    duplicate_timestamp_count = int(parsed_timestamp.duplicated().sum())
    if duplicate_timestamp_count:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} has "
            f"{duplicate_timestamp_count} duplicate values."
        )

    missing_values = {column: int(df[column].isna().sum()) for column in df.columns}
    dtypes = {column: str(dtype) for column, dtype in df.dtypes.items()}
    schema_summary = {
        "columns": list(df.columns),
        "dtypes": dtypes,
        "missing_values": missing_values,
        "row_count": len(df),
    }
    return SchemaValidationReport(
        row_count=len(df),
        columns=tuple(str(column) for column in df.columns),
        dtypes=dtypes,
        missing_values=missing_values,
        timestamp_column=timestamp_column,
        duplicate_timestamp_count=duplicate_timestamp_count,
        schema_summary=schema_summary,
    )


def _dtype_matches(series: pd.Series, expected: str) -> bool:
    normalized = expected.lower()
    if normalized in {"numeric", "number"}:
        return bool(is_numeric_dtype(series))
    if normalized in {"datetime", "timestamp", "datetime64"}:
        return bool(is_datetime64_any_dtype(series))
    if normalized in {"string", "str", "text"}:
        return bool(is_string_dtype(series) or series.dtype == object)
    if normalized in {"bool", "boolean"}:
        return bool(is_bool_dtype(series))
    return str(series.dtype) == expected
