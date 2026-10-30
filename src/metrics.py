"""Forecast error metrics (raw kW scale)."""
import numpy as np


def _flat(y_true, y_pred):
    return np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()


def mae(y_true, y_pred):
    a, b = _flat(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred):
    a, b = _flat(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred):
    a, b = _flat(y_true, y_pred)
    m = a != 0
    return float(np.mean(np.abs((a[m] - b[m]) / a[m])) * 100.0)


def r2(y_true, y_pred):
    a, b = _flat(y_true, y_pred)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def all_metrics(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred), "R2": r2(y_true, y_pred)}
