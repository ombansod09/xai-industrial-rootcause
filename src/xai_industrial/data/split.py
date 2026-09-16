from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    *,
    timestamp_column: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float | None = None,
    gap_seconds: int = 0,
) -> TimeSplit:
    _validate_split_args(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        gap_seconds=gap_seconds,
    )
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column {timestamp_column!r} is missing.")
    if len(df) < 3:
        raise ValueError("At least three rows are required for train/validation/test split.")

    out = df.copy()
    timestamps = pd.to_datetime(out[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Timestamp column {timestamp_column!r} contains null values.")
    if timestamps.duplicated().any():
        raise ValueError(f"Timestamp column {timestamp_column!r} contains duplicate values.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            f"Timestamp column {timestamp_column!r} must be sorted in chronological order."
        )
    out[timestamp_column] = timestamps

    n = len(df)
    i = int(n * train_fraction)
    j = int(n * (train_fraction + validation_fraction))
    if i <= 0 or j <= i or j >= n:
        raise ValueError("Split fractions produce an empty split.")

    train = out.iloc[:i].copy()
    validation = out.iloc[i:j].copy()
    test = out.iloc[j:].copy()

    gap = pd.Timedelta(seconds=gap_seconds)
    if gap_seconds:
        validation = validation[
            validation[timestamp_column] >= train[timestamp_column].max() + gap
        ].copy()
        if validation.empty:
            raise ValueError("Train/validation gap removes all validation rows.")
        test = test[test[timestamp_column] >= validation[timestamp_column].max() + gap].copy()
        if test.empty:
            raise ValueError("Validation/test gap removes all test rows.")

    _assert_order_and_gaps(
        train,
        validation,
        test,
        timestamp_column=timestamp_column,
        gap_seconds=gap_seconds,
    )
    return TimeSplit(train=train, validation=validation, test=test)


def _validate_split_args(
    *,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float | None,
    gap_seconds: int,
) -> None:
    if test_fraction is None:
        test_fraction = 1.0 - train_fraction - validation_fraction
    fractions = [train_fraction, validation_fraction, test_fraction]
    if any(f <= 0.0 or f >= 1.0 for f in fractions):
        raise ValueError("Split fractions must each be greater than 0 and less than 1.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be non-negative.")


def _assert_order_and_gaps(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    timestamp_column: str,
    gap_seconds: int,
) -> None:
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must be non-empty.")
    gap = pd.Timedelta(seconds=gap_seconds)
    if train[timestamp_column].max() + gap > validation[timestamp_column].min():
        raise ValueError("Train/validation gap is smaller than requested.")
    if validation[timestamp_column].max() + gap > test[timestamp_column].min():
        raise ValueError("Validation/test gap is smaller than requested.")
