# src/utils.py

import numpy as np

#tankes an int seed as input 
def set_global_seed(seed: int) -> np.random.Generator:
    """
    Create and return a NumPy random Generator with a fixed seed.
    Useful so runs are reproducible.
    """
    return np.random.default_rng(seed) #creates a new randoome generator with given seed, which can be used to create randome numbers in reproducible way 


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float: #takes two arrays, actual values, and predicted values
    """
    Mean Absolute Error for regression.
    y_true, y_pred: shape (n_samples, 1) or (n_samples,)
    """
    y_true = np.asarray(y_true).reshape(-1) #reshape -1, flatten arrays into 1D vectors, 
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(np.abs(y_true - y_pred))) #returns a float, MAE. 
#calcultaes diff between actual and predicted
#then takes absolute value 
#then mean 
#and covnert to float 
