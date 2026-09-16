import pandas as pd
import pytest

from xai_industrial.data.split import chronological_split


def _frame(periods=120, freq="10s"):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=periods, freq=freq),
            "value": range(periods),
        }
    )


def test_split_is_chronological():
    df = _frame()
    s = chronological_split(
        df,
        timestamp_column="timestamp",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
    )
    assert s.train["timestamp"].max() < s.validation["timestamp"].min()
    assert s.validation["timestamp"].max() < s.test["timestamp"].min()


def test_unsorted_timestamps_fail():
    df = _frame()
    df.loc[3, "timestamp"] = df.loc[0, "timestamp"] - pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="chronological order"):
        chronological_split(df, timestamp_column="timestamp")


def test_null_timestamps_fail():
    df = _frame()
    df.loc[3, "timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="null"):
        chronological_split(df, timestamp_column="timestamp")


def test_duplicate_timestamps_fail():
    df = _frame()
    df.loc[3, "timestamp"] = df.loc[2, "timestamp"]
    with pytest.raises(ValueError, match="duplicate"):
        chronological_split(df, timestamp_column="timestamp")


def test_invalid_split_fractions_fail():
    df = _frame()
    with pytest.raises(ValueError, match="sum to 1.0"):
        chronological_split(
            df,
            timestamp_column="timestamp",
            train_fraction=0.5,
            validation_fraction=0.2,
            test_fraction=0.2,
        )


def test_train_validation_gap_uses_timestamps():
    df = _frame(periods=180, freq="10s")
    s = chronological_split(
        df,
        timestamp_column="timestamp",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        gap_seconds=300,
    )
    observed_gap = s.validation["timestamp"].min() - s.train["timestamp"].max()
    assert observed_gap >= pd.Timedelta(seconds=300)


def test_validation_test_gap_uses_timestamps():
    df = _frame(periods=220, freq="10s")
    s = chronological_split(
        df,
        timestamp_column="timestamp",
        train_fraction=0.4,
        validation_fraction=0.3,
        test_fraction=0.3,
        gap_seconds=300,
    )
    observed_gap = s.test["timestamp"].min() - s.validation["timestamp"].max()
    assert observed_gap >= pd.Timedelta(seconds=300)


def test_boundary_conditions_fail_when_split_empty():
    df = _frame(periods=3)
    with pytest.raises(ValueError, match="empty split"):
        chronological_split(
            df,
            timestamp_column="timestamp",
            train_fraction=0.8,
            validation_fraction=0.1,
            test_fraction=0.1,
        )


def test_split_is_reproducible():
    df = _frame()
    first = chronological_split(df, timestamp_column="timestamp")
    second = chronological_split(df, timestamp_column="timestamp")
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)
