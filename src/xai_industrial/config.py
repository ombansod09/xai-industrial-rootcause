from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str
    raw_path: Path
    processed_path: Path
    timestamp_column: str


@dataclass(frozen=True)
class SplitConfig:
    strategy: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    gap_seconds: int


@dataclass(frozen=True)
class WindowConfig:
    length: int
    stride: int


@dataclass(frozen=True)
class XaiConfig:
    top_k: int
    perturbation_repeats: int
    perturbation_scale: float


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_repeats: int
    random_seeds: tuple[int, ...]


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    data: DataConfig
    split: SplitConfig
    window: WindowConfig
    xai: XaiConfig
    evaluation: EvaluationConfig


def load_config(path: str | Path = "configs/base.yaml") -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError("Configuration file must contain a mapping.")
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> ProjectConfig:
    seed = _required_int(raw, "seed", minimum=0)
    data_raw = _required_mapping(raw, "data")
    split_raw = _required_mapping(raw, "split")
    window_raw = _required_mapping(raw, "window")
    xai_raw = _required_mapping(raw, "xai")
    evaluation_raw = _required_mapping(raw, "evaluation")

    data = DataConfig(
        dataset_name=_required_str(data_raw, "dataset_name"),
        raw_path=Path(_required_str(data_raw, "raw_path")),
        processed_path=Path(_required_str(data_raw, "processed_path")),
        timestamp_column=_required_str(data_raw, "timestamp_column"),
    )
    split = SplitConfig(
        strategy=_required_str(split_raw, "strategy"),
        train_fraction=_required_float(split_raw, "train_fraction"),
        validation_fraction=_required_float(split_raw, "validation_fraction"),
        test_fraction=_required_float(split_raw, "test_fraction"),
        gap_seconds=_required_int(split_raw, "gap_seconds", minimum=0),
    )
    window = WindowConfig(
        length=_required_int(window_raw, "length", minimum=1),
        stride=_required_int(window_raw, "stride", minimum=1),
    )
    xai = XaiConfig(
        top_k=_required_int(xai_raw, "top_k", minimum=1),
        perturbation_repeats=_required_int(xai_raw, "perturbation_repeats", minimum=1),
        perturbation_scale=_required_float(xai_raw, "perturbation_scale", minimum=0.0),
    )
    evaluation = EvaluationConfig(
        bootstrap_repeats=_required_int(evaluation_raw, "bootstrap_repeats", minimum=1),
        random_seeds=tuple(_required_int_list(evaluation_raw, "random_seeds", minimum=0)),
    )

    _validate_split(split)
    return ProjectConfig(seed, data, split, window, xai, evaluation)


def _validate_split(split: SplitConfig) -> None:
    if split.strategy != "chronological":
        raise ValueError("Only chronological split strategy is supported.")
    fractions = [split.train_fraction, split.validation_fraction, split.test_fraction]
    if any(f <= 0.0 or f >= 1.0 for f in fractions):
        raise ValueError("Split fractions must each be greater than 0 and less than 1.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0.")


def _required_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"Configuration key {key!r} must be a mapping.")
    return value


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Configuration key {key!r} must be a non-empty string.")
    return value


def _required_int(raw: dict[str, Any], key: str, minimum: int | None = None) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"Configuration key {key!r} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration key {key!r} must be >= {minimum}.")
    return value


def _required_float(raw: dict[str, Any], key: str, minimum: float | None = None) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Configuration key {key!r} must be numeric.")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"Configuration key {key!r} must be >= {minimum}.")
    return value


def _required_int_list(
    raw: dict[str, Any], key: str, minimum: int | None = None
) -> list[int]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Configuration key {key!r} must be a non-empty list.")
    out = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise TypeError(f"Configuration key {key!r} must contain only integers.")
        if minimum is not None and item < minimum:
            raise ValueError(f"Configuration key {key!r} values must be >= {minimum}.")
        out.append(item)
    return out
