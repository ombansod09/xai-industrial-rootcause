
import numpy as np
import pytest

from xai_industrial.xai.stability import evaluate_ranking_stability


def sample_data():
    training = np.array([
        [1.0, 10.0, 100.0],
        [2.0, 11.0, 101.0],
        [3.0, 12.0, 102.0],
        [4.0, 13.0, 103.0],
        [5.0, 14.0, 104.0],
    ])
    observation = np.array([20.0, 11.0, 100.0])
    return observation, training, ["pressure", "temperature", "flow"]


def test_stability_is_repeatable_for_same_seed():
    observation, training, names = sample_data()
    kwargs = dict(repeats=8, perturbation_scale=0.05, seed=123)

    first = evaluate_ranking_stability(observation, training, names, **kwargs)
    second = evaluate_ranking_stability(observation, training, names, **kwargs)

    assert first == second


def test_zero_perturbation_produces_perfect_stability():
    observation, training, names = sample_data()

    result = evaluate_ranking_stability(
        observation, training, names, repeats=5, perturbation_scale=0.0
    )

    assert result["top_k_jaccard_stability"] == 1.0


def test_top_k_is_capped_by_sensor_count():
    observation, training, names = sample_data()

    result = evaluate_ranking_stability(
        observation, training, names, repeats=4, top_k=5
    )

    assert result["top_k"] == 3
    assert 0.0 <= result["top_k_jaccard_stability"] <= 1.0


@pytest.mark.parametrize("repeats", [0, 1])
def test_rejects_too_few_repeats(repeats):
    observation, training, names = sample_data()

    with pytest.raises(ValueError, match="repeats"):
        evaluate_ranking_stability(
            observation, training, names, repeats=repeats
        )


def test_rejects_negative_perturbation_scale():
    observation, training, names = sample_data()

    with pytest.raises(ValueError, match="perturbation_scale"):
        evaluate_ranking_stability(
            observation, training, names, perturbation_scale=-0.1
        )


def test_rejects_mismatched_sensor_names():
    observation, training, _ = sample_data()

    with pytest.raises(ValueError, match="sensor_names"):
        evaluate_ranking_stability(observation, training, ["pressure"])
