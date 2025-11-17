# src/run_experiments.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ann import ANN
from pso import PSO
from utils import mae
from main import load_concrete_dataset   # reuse your loader


# ---------------------------------------------------------
# EXPERIMENT CONFIGURATIONS
# ---------------------------------------------------------
"""architectures = {
    "shallow":  ([8, 8, 1], ["relu"]),                # 1 hidden layer
    "medium":   ([8, 16, 8, 1], ["relu", "relu"]),    # 2 hidden layers
    "deep":     ([8, 16, 16, 8, 1], ["relu", "relu", "relu"]),  # 3 hidden layers
}
activation_functions = ["relu", "tanh", "sigmoid"]
"""
architectures = {
    "shallow": ([8, 20, 1], ["relu"]),
    "medium":  ([8, 32, 16, 1], ["relu", "relu"]),
    "deep":    ([8, 32, 16, 8, 1], ["relu", "relu", "relu"]),
}


activation_functions = ["relu", "tanh", "sigmoid"]


# Ensure folders exist
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------
# RUN SINGLE EXPERIMENT
# ---------------------------------------------------------
def run_single_experiment(arch_name, layer_sizes, base_activations, act_fn, X_train, y_train, X_test, y_test):
    """
    Runs one experiment: specific architecture + activation function.
    Returns final train MAE, test MAE, and PSO history.
    """

    # Replace all hidden layer activations with chosen act_fn
    activations = [act_fn] * (len(layer_sizes) - 2)   # hidden layers only

    # Build model
    ann = ANN(layer_sizes, activations)
    dim = ann.num_params()

    # Build PSO fitness function
    def fitness_fn(vec):
        ann.set_param_vector(vec)
        preds = ann.forward(X_train)
        return -mae(y_train, preds)

# WE CHANGED OUR PARAMS TO MAKE MAE AND RESULTS BETTER.. 
    """# PSO definition
    bounds = (-1.0, 1.0)
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
    ) """

    pso = PSO(
    dim=dim,
    fitness_fn=fitness_fn,
    bounds=(-1, 1),
    swarm_size=50,      # increased from 30 → 50
    alpha=0.7,          # lower inertia (was 0.9)
    beta=0.3,           # stronger personal influence (was 0.1)
    gamma=0.3,          # stronger informants influence (was 0.1)
    delta=0.1,          # small global influence (was 0.0)
    e=1.0,
    n_informants=7,     # more informants (was 5)
    rng_seed=123,
)

    # Run optimisation
    # best_vec, best_fit, history = pso.run(max_iter=50, verbose=False)

    best_vec, best_fit, history = pso.run(max_iter=150, verbose=False)


    # Evaluate on test set
    ann.set_param_vector(best_vec)
    test_preds = ann.forward(X_test)
    test_mae = mae(y_test, test_preds)

    return -best_fit, test_mae, history  # convert train fitness → MAE


# ---------------------------------------------------------
# MAIN EXPERIMENT LOOP
# ---------------------------------------------------------
def main():

    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_concrete_dataset()

    results = []

    print("\nRunning full experiment grid...\n")

    for arch_name, (layer_sizes, base_acts) in architectures.items():
        for act_fn in activation_functions:

            print(f"→ Running {arch_name.upper()} with {act_fn.upper()} activation...")

            train_mae, test_mae, history = run_single_experiment(
                arch_name,
                layer_sizes,
                base_acts,
                act_fn,
                X_train,
                y_train,
                X_test,
                y_test
            )

            # Save plot of convergence curve
            plt.figure(figsize=(6, 4))
            plt.plot(history)
            plt.xlabel("Iteration")
            plt.ylabel("Best Fitness (Negative MAE)")
            plt.title(f"PSO Convergence — {arch_name} — {act_fn}")
            plt.grid(True)

            plot_path = os.path.join(PLOTS_DIR, f"{arch_name}_{act_fn}_convergence.png")
            plt.savefig(plot_path, dpi=150)
            plt.close()

            # Store results
            results.append({
                "Architecture": arch_name,
                "Activation": act_fn,
                "Train MAE": train_mae,
                "Test MAE": test_mae,
                "Convergence Plot": plot_path,
            })

    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    df.to_csv(csv_path, index=False)

    print("\nAll experiments completed!")
    print("Results saved to:", csv_path)
    print("\nSummary:\n")
    print(df)


if __name__ == "__main__":
    main()
