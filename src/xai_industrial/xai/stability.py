import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))
