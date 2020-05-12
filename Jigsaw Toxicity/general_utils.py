import numpy as np


def false_positive_rate(y_true, y_pred):
    tp = np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    fp = np.sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
    tn = np.sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0])
    fn = np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])
    return fp / (fp + tn)


def false_negative_rate(y_true, y_pred):
    tp = np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    fp = np.sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
    tn = np.sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 0])
    fn = np.sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0])
    return fn / (fn + tp)
