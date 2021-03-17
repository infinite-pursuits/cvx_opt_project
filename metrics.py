import numpy as np


def restore(y, scaler):
    return scaler.scale_ * (y + scaler.mean_)


def MAE(target, rec, scaler=None):
    if scaler is not None:
        target = restore(target, scaler)
        rec = restore(rec, scaler)
    return np.abs(target-rec).mean()


def absmin(target, rec, scaler=None):
    if scaler is not None:
        target = restore(target, scaler)
        rec = restore(rec, scaler)
    return np.min(np.abs(target-rec))


def absmax(target, rec, scaler=None):
    if scaler is not None:
        target = restore(target, scaler)
        rec = restore(rec, scaler)
    return np.max(np.abs(target-rec))
