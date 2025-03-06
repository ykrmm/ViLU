import numpy as np
import sklearn.metrics as skm
from scipy import interpolate


def get_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    data = np.concatenate((pos, neg))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_det_accuracy(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    data = np.concatenate((pos, neg))
    fpr, tpr, _ = skm.roc_curve(labels, data)
    return 0.5 * (tpr + 1.0 - fpr).max()


def get_aupr_out(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [0] * len(pos) + [1] * len(neg)
    data = np.concatenate((pos, neg))
    aupr_out = skm.average_precision_score(
        labels, data
    )  # data = probability estimates of the positive class
    return aupr_out


def get_aupr_in(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    data = np.concatenate((pos, neg))
    aupr_in = skm.average_precision_score(labels, data)
    return aupr_in


def _get_fpr(pos: np.ndarray, neg: np.ndarray) -> float:
    return np.sum(neg < np.percentile(pos, 95)) / len(neg)


def get_fpr(pos: np.ndarray, neg: np.ndarray) -> float:
    labels = [1] * len(pos) + [0] * len(neg)
    data = np.concatenate((pos, neg))
    fpr, tpr, _ = skm.roc_curve(
        labels, data
    )  # data must be prob estimates / conf values of positive class
    return float(interpolate.interp1d(tpr, fpr)(0.95))


def get_oscr(
    pos: np.ndarray,
    neg: np.ndarray,
    pred: np.ndarray,
    y_ind: np.ndarray,
) -> float:
    score = np.concatenate((pos, neg), axis=0)

    def get_fpr(t):  # noqa ANN201
        return (neg >= t).sum() / len(neg)

    def get_ccr(t):  # noqa ANN201
        return ((pos > t) & (pred == y_ind)).sum() / len(pos)

    fpr = [0.0]
    ccr = [0.0]
    for s in -np.sort(-score):
        fpr.append(get_fpr(s))
        ccr.append(get_ccr(s))
    fpr.append(1.0)
    ccr.append(1.0)
    roc = sorted(zip(fpr, ccr), reverse=True)
    oscr = 0.0
    for i in range(len(score)):
        oscr += (roc[i][0] - roc[i + 1][0]) * (roc[i][1] + roc[i + 1][1]) / 2.0
    return oscr


def get_accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    return np.mean(pred == labels)
