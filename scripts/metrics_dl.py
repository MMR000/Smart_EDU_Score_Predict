import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def recall_at_topk(y_true, y_score, k_frac=0.05):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    k = max(1, int(round(n * k_frac)))
    idx = np.argsort(-y_score)[:k]
    tp = int(y_true[idx].sum())
    denom = int(y_true.sum()) if int(y_true.sum()) > 0 else 1
    return float(tp / denom)

def classification_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "PosRate": float(y_true.mean()),
        "Brier": float(brier_score_loss(y_true, y_prob)),
    }

    # AUCs can fail if only one class in y_true (rare but possible on tiny splits)
    try:
        out["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["ROC_AUC"] = float("nan")

    try:
        out["PR_AUC"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        out["PR_AUC"] = float("nan")

    out["Recall@Top5%"] = recall_at_topk(y_true, y_prob, 0.05)
    out["Recall@Top10%"] = recall_at_topk(y_true, y_prob, 0.10)
    return out
