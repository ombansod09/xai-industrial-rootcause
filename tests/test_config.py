import pytest

from xai_industrial.config import parse_config


def _config():
    return {
        "seed": 42,
        "data": {
            "dataset_name": "metropt",
            "raw_path": "data/raw/dataset_train.csv",
            "processed_path": "data/processed/metropt.parquet",
            "timestamp_column": "timestamp",
        },
        "split": {
            "strategy": "chronological",
            "train_fraction": 0.7,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "gap_seconds": 300,
        },
        "window": {"length": 60, "stride": 10},
        "xai": {
            "top_k": 5,
            "perturbation_repeats": 30,
            "perturbation_scale": 0.01,
        },
        "evaluation": {"bootstrap_repeats": 1000, "random_seeds": [13, 42]},
    }


def test_parse_config_validates_base_shape():
    config = parse_config(_config())
    assert config.seed == 42
    assert config.data.timestamp_column == "timestamp"
    assert config.split.strategy == "chronological"


def test_parse_config_rejects_invalid_fractions():
    raw = _config()
    raw["split"]["test_fraction"] = 0.2
    with pytest.raises(ValueError, match="sum to 1.0"):
        parse_config(raw)
