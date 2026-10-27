
"""Chronological train/validation/test splitting with optional purge gaps."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    purge_rows: int = 0,
) -> TimeSplit:
    """Split rows chronologically, excluding rows around partition boundaries.

    ``purge_rows`` rows are excluded after the training boundary and after
    the validation boundary. Fractions determine the original boundaries
    before purging. The input is assumed to be chronologically ordered.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    if isinstance(purge_rows, bool) or not isinstance(purge_rows, int):
        raise ValueError("purge_rows must be a non-negative integer")
    if purge_rows < 0:
        raise ValueError("purge_rows must be a non-negative integer")

    n = len(df)
    i = int(n * train_fraction)
    j = int(n * (train_fraction + validation_fraction))

    train = df.iloc[:i].copy()
    validation = df.iloc[i + purge_rows:j].copy()
    test = df.iloc[j + purge_rows:].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Split produced an empty partition. Reduce purge_rows "
            "or provide more rows."
        )

    return TimeSplit(train=train, validation=validation, test=test)
