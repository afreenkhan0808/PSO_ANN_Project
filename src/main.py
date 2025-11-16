# src/main.py

import numpy as np
import pandas as pd

from ann import ANN
from pso import PSO
from utils import mae


# ---------------------------------------------------------
# 1) Load dataset
# ---------------------------------------------------------
def load_concrete_dataset():
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go from src/ → project root
    CSV_PATH = os.path.join(BASE_DIR, "data", "concrete_data.csv")
    df = pd.read_csv(CSV_PATH)

    
    X = df.iloc[:, :-1].values.astype(float)   # first 8 columns
    y = df.iloc[:, -1].values.astype(float).reshape(-1, 1)  # last column

    # Shuffle
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Train/test split (70/30)
    n_train = int(0.7 * len(X))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Normalise inputs using training stats
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # For regression you can either normalise y or keep as-is.
    # Let’s keep y unscaled (easier interpretability).
    return X_train_norm, X_test_norm, y_train, y_test


# ---------------------------------------------------------
# 2) Fitness function wrapper for PSO
# ---------------------------------------------------------
def make_fitness_fn(ann, X_train, y_train):
    def fitness_fn(weight_vector):
        ann.set_param_vector(weight_vector)
        preds = ann.forward(X_train)

        # Compute MAE (smaller is better),
        # but PSO maximises fitness → return negative MAE.
        return -mae(y_train, preds)
    return fitness_fn


# ---------------------------------------------------------
# 3) MAIN PROGRAM — Train ANN using PSO
# ---------------------------------------------------------
def main():

    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_concrete_dataset()

    # ANN architecture (you can test different ones later!)
    layer_sizes = [8, 16, 8, 1]               # 8 → h1=16 → h2=8 → output=1
    activations = ["relu", "relu"]            # one per hidden layer

    ann = ANN(layer_sizes, activations)
    dim = ann.num_params()                    # size of PSO particle vector

    print(f"ANN total parameters: {dim}")

    # Build fitness function for PSO
    fitness_fn = make_fitness_fn(ann, X_train, y_train)

    # PSO bounds for weights (you can tune this later)
    bounds = (-1.0, 1.0)

    # Create PSO instance
    pso = PSO(
        dim=dim,
        fitness_fn=fitness_fn,
        bounds=bounds,
        swarm_size=30,
        alpha=0.9,
        beta=0.1,
        gamma=0.1,
        delta=0.0,
        e=1.0,
        n_informants=5,
        rng_seed=123,
    )

    print("Running PSO optimisation...")
    best_vec, best_fit, history = pso.run(max_iter=50, verbose=True)

    print("\nPSO training complete.")
    print("Best training fitness (MAE):", best_fit)

    # Evaluate ANN using best found weights
    ann.set_param_vector(best_vec)
    preds = ann.forward(X_test)
    test_mae = mae(y_test, preds)

    print("Test MAE:", test_mae)
    print("\nDone.")


if __name__ == "__main__":
    main()
