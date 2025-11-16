# src/utils.py

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """
    Create and return a NumPy random Generator with a fixed seed.
    Useful so runs are reproducible.
    """
    return np.random.default_rng(seed)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error for regression.
    y_true, y_pred: shape (n_samples, 1) or (n_samples,)
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred)))
