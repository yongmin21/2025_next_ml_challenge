import numpy as np
import sklearn.metrics

def weighted_logloss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray = 1e-15,
) -> float:
    y_pred = np.clip(y_pred, weight, 1 - weight)
    
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    
    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0
    
    return 0.5 * ll_0 + 0.5 * ll_1
    
def competition_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    ap = sklearn.metrics.average_precision_score(y_true, y_pred)
    wll = weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll