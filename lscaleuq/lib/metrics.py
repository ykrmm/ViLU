import numpy as np
import sklearn.metrics as skm
from scipy import interpolate


def get_auroc(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((xin, xood))
    auroc = skm.roc_auc_score(labels, data)
    return auroc


def get_det_accuracy(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((xin, xood))
    fpr, tpr, _ = skm.roc_curve(labels, data)
    return 0.5 * (tpr + 1.0 - fpr).max()


def get_aupr_out(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [0] * len(xin) + [1] * len(xood)
    data = np.concatenate((xin, xood))
    aupr_out = skm.average_precision_score(
        labels, data
    )  # data = probability estimates of the positive class
    return aupr_out


def get_aupr_in(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((xin, xood))
    aupr_in = skm.average_precision_score(labels, data)
    return aupr_in


def _get_fpr(xin: np.ndarray, xood: np.ndarray) -> float:
    return np.sum(xood < np.percentile(xin, 95)) / len(xood)


def get_fpr(xin: np.ndarray, xood: np.ndarray) -> float:
    labels = [1] * len(xin) + [0] * len(xood)
    data = np.concatenate((xin, xood))
    fpr, tpr, _ = skm.roc_curve(
        labels, data
    )  # data must be prob estimates / conf values of positive class
    return float(interpolate.interp1d(tpr, fpr)(0.95))


def get_oscr(
    xin: np.ndarray,
    xood: np.ndarray,
    pred: np.ndarray,
    y_ind: np.ndarray,
) -> float:
    score = np.concatenate((xin, xood), axis=0)

    def get_fpr(t):  # noqa ANN201
        return (xood >= t).sum() / len(xood)

    def get_ccr(t):  # noqa ANN201
        return ((xin > t) & (pred == y_ind)).sum() / len(xin)

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
