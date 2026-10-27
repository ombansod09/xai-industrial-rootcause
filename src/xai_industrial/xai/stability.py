import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


"""Perturbation-based stability for deviation-ranking explanations."""

from itertools import combinations

import numpy as np

from xai_industrial.xai.agreement import jaccard


def evaluate_ranking_stability(
    observation: np.ndarray,
    training_data: np.ndarray,
    sensor_names: list[str],
    *,
    repeats: int = 20,
    perturbation_scale: float = 0.01,
    seed: int = 42,
    top_k: int = 5,
) -> dict[str, float | int]:
    """Measure top-k ranking consistency under Gaussian sensor perturbations.

    Perturbation scale is a multiplier of each sensor's training standard
    deviation. This evaluates deviation-ranking consistency, not causality,
    detector prediction stability, or explanation faithfulness.
    """
    observation = np.asarray(observation, dtype=float)
    training_data = np.asarray(training_data, dtype=float)

    if training_data.ndim != 2:
        raise ValueError("training_data must be a 2D array.")
    if observation.ndim != 1 or observation.size != training_data.shape[1]:
        raise ValueError("observation must match the number of training features.")
    if len(sensor_names) != observation.size:
        raise ValueError("sensor_names must match the number of features.")
    if not np.isfinite(observation).all() or not np.isfinite(training_data).all():
        raise ValueError("observation and training_data must be finite.")
    if training_data.shape[0] < 1:
        raise ValueError("training_data must contain at least one row.")
    if repeats < 2:
        raise ValueError("repeats must be at least 2.")
    if not np.isfinite(perturbation_scale) or perturbation_scale < 0:
        raise ValueError("perturbation_scale must be finite and non-negative.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    k = min(int(top_k), len(sensor_names))
    if k == 0:
        raise ValueError("At least one sensor is required.")

    center = np.nanmedian(training_data, axis=0)
    mad = np.nanmedian(np.abs(training_data - center), axis=0) * 1.4826
    q75, q25 = np.nanpercentile(training_data, [75, 25], axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.nanstd(training_data, axis=0)

    robust_scale = np.where(
        mad > 1e-9,
        mad,
        np.where(iqr_scale > 1e-9, iqr_scale, std_scale),
    )
    robust_scale = np.where(
        np.isfinite(robust_scale) & (robust_scale > 1e-9),
        robust_scale,
        1.0,
    )

    noise_scale = np.std(training_data, axis=0)
    noise_scale = np.where(
        np.isfinite(noise_scale) & (noise_scale > 1e-9),
        noise_scale,
        1.0,
    )

    rng = np.random.default_rng(seed)
    rankings: list[set[str]] = []

    for _ in range(repeats):
        perturbed = observation + rng.normal(
            loc=0.0,
            scale=perturbation_scale,
            size=observation.size,
        ) * noise_scale

        deviations = np.abs((perturbed - center) / robust_scale)
        top_indices = np.argsort(-deviations, kind="stable")[:k]
        rankings.append({sensor_names[i] for i in top_indices})

    pairwise = [jaccard(a, b) for a, b in combinations(rankings, 2)]

    return {
        "top_k_jaccard_stability": float(np.mean(pairwise)),
        "stability_repeats": int(repeats),
        "top_k": int(k),
    }
