def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return 1.0 if not union else len(a & b) / len(union)
