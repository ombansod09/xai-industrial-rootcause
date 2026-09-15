from sklearn.metrics import average_precision_score, precision_recall_fscore_support

def binary_metrics(y_true, y_pred, y_score=None):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    out = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    if y_score is not None:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    return out
