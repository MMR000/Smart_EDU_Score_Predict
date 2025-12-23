import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    precision_recall_curve, roc_curve, brier_score_loss
)

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "PosRate": float(y_true.mean()),
        "Brier": float(brier_score_loss(y_true, y_prob)),
    }
    if len(np.unique(y_true)) > 1:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
        out["PR_AUC"] = float(average_precision_score(y_true, y_prob))
    else:
        out["ROC_AUC"] = None
        out["PR_AUC"] = None
    return out

def recall_at_topk(y_true, y_prob, topk_percent=0.05):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    k = max(1, int(round(n * float(topk_percent))))
    idx = np.argsort(-y_prob)[:k]
    total_pos = max(1, int(y_true.sum()))
    captured = int(y_true[idx].sum())
    return {"TopK%": float(topk_percent), "TopK": int(k), "Recall@TopK": float(captured / total_pos)}
