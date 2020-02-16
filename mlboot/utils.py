"""
Here we define a list of commonly-used pointwise metrics
"""
import numpy as np


def pointwise_binary_score(y_pred, y_true):
    """
    Elementwise 0-1 score. Pointwise equivalent of accuracy score.
    """
    return y_pred == y_true


def pointwise_mae(y_pred, y_true):
    return np.abs(y_pred - y_true)


def pointwise_mse(y_pred, y_true):
    return (y_pred - y_true) ** 2


def pointwise_rmse(y_pred, y_true):
    return pointwise_mse(y_pred, y_true) ** 0.5
