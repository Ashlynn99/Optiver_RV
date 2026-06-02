import numpy as np


def rmspe(y_true, y_pred):
    """Root mean squared percentage error, matching the competition metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(y_true), 1e-12)
    return np.sqrt(np.mean(np.square((y_true - y_pred) / denominator)))


def realized_volatility(series):
    x = series.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    return np.sqrt(np.sum(x ** 2))


def realized_absvar(series):
    x = series.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return np.nan
    return np.sum(np.abs(x))


def realized_skew(series):
    x = series.dropna().to_numpy(dtype=float)
    if len(x) < 3:
        return 0.0
    rv = np.sqrt(np.sum(x ** 2))
    if rv == 0:
        return 0.0
    return np.sum(x ** 3) / (rv ** 3)


def realized_kurtosis(series):
    x = series.dropna().to_numpy(dtype=float)
    if len(x) < 4:
        return 0.0
    rv = np.sqrt(np.sum(x ** 2))
    if rv == 0:
        return 0.0
    return np.sum(x ** 4) / (rv ** 4)

