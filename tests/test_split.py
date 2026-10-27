
import pandas as pd
import pytest

from xai_industrial.data.split import chronological_split


def make_frame(n=100):
    return pd.DataFrame({
        "t": pd.date_range("2026-01-01", periods=n, freq="s"),
        "value": range(n),
    })


def test_split_is_chronological():
    split = chronological_split(make_frame())

    assert split.train["t"].max() < split.validation["t"].min()
    assert split.validation["t"].max() < split.test["t"].min()


def test_default_split_preserves_original_boundaries():
    split = chronological_split(make_frame())

    assert len(split.train) == 70
    assert len(split.validation) == 15
    assert len(split.test) == 15


def test_purge_gap_excludes_rows_at_both_boundaries():
    frame = make_frame()
    split = chronological_split(frame, purge_rows=2)

    assert split.train["value"].tolist() == list(range(70))
    assert split.validation["value"].tolist() == list(range(72, 85))
    assert split.test["value"].tolist() == list(range(87, 100))

    # Rows 70-71 and 85-86 are purged.
    retained = set(split.train["value"]) | set(split.validation["value"]) | set(split.test["value"])
    assert retained.isdisjoint({70, 71, 85, 86})


@pytest.mark.parametrize("purge_rows", [-1, 1.5, True])
def test_invalid_purge_rows_rejected(purge_rows):
    with pytest.raises(ValueError, match="purge_rows"):
        chronological_split(make_frame(), purge_rows=purge_rows)


def test_purge_gap_cannot_empty_a_partition():
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split(make_frame(), purge_rows=16)


def test_invalid_fractions_rejected():
    with pytest.raises(ValueError):
        chronological_split(make_frame(), train_fraction=0)

    with pytest.raises(ValueError):
        chronological_split(
            make_frame(),
            train_fraction=0.8,
            validation_fraction=0.2,
        )
