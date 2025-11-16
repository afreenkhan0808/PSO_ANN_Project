import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from ann import ANN
from pso import PSO
from utils import mae


# ---------------------------------------------------------
# Load dataset (same code as main.py, re-used here)
# ---------------------------------------------------------
def load_concrete_dataset():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "data", "concrete_data.csv")

    df = pd.read_csv(CSV_PATH)

    X = df.iloc[:, :-1].values.astype(float) #splitting data into features x and target y 
    y = df.iloc[:, -1].values.astype(float).reshape(-1, 1)

    rng = np.random.default_rng(42) #shuffles data randomly 
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    n_train = int(0.7 * len(X)) #splits into 70% training and 30% test 
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    X_mean = X_train.mean(axis=0) #normalising features to zero mean 
    X_std = X_train.std(axis=0) + 1e-8 #unit variance 

    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    return X_train_norm, X_test_norm, y_train, y_test #return normalized training and test data 


# ---------------------------------------------------------
# Fitness function generator
# ---------------------------------------------------------
def make_fitness_fn(ann, X_train, y_train): #wraps the ann evalutation in a fuction PSO can optimize 
    def fitness_fn(vec):
        ann.set_param_vector(vec) #takes a weight vector from PSO, set ANN parameters
        preds = ann.forward(X_train) #predicts output
        return -mae(y_train, preds) #return negative MAE bc PSO maximies fitness 
    return fitness_fn


# ---------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------

ANN_ARCHITECTURES = [
    ([8, 8, 1], ["relu"]),                  # simple
    ([8, 16, 8, 1], ["relu", "relu"]),      # medium our default 
    ([8, 32, 16, 8, 1], ["relu", "relu", "relu"]),  # deeper
]

PSO_SETTINGS = [
    # swarm_size, iterations, alpha, beta, gamma
    (10,  50, 0.9, 0.1, 0.1),
    (25,  20, 0.9, 0.1, 0.1),
    (50,  10, 0.9, 0.1, 0.1),

    # coefficient variations
    (30, 50, 0.7, 0.3, 0.1),
    (30, 50, 0.7, 0.1, 0.2),
]


# ---------------------------------------------------------
# Run a single experiment configuration (10 runs)
# ---------------------------------------------------------
def run_single_configuration(layer_sizes, activations, swarm, iters, alpha, beta, gamma):
    X_train, X_test, y_train, y_test = load_concrete_dataset()

    test_maes = [] #list to store MAES from 10 runs 

    for seed in range(10):   # rpeating ten times for different seeds , getting avg performance 
        ann = ANN(layer_sizes, activations) #initialising ann and its functions 
        dim = ann.num_params()
        fitness_fn = make_fitness_fn(ann, X_train, y_train)

        pso = PSO( #initilise PSO with given configuration and reurtn BESS ANN WEIGHTS 
            dim=dim,
            fitness_fn=fitness_fn,
            bounds=(-1, 1),
            swarm_size=swarm,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=0.0,
            e=1.0,
            n_informants=5,
            rng_seed=seed,
        )

        best_vec, best_fit, history = pso.run(max_iter=iters, verbose=False)

        # Evaluate on test set and store MAE for this run 
        ann.set_param_vector(best_vec)
        preds = ann.forward(X_test)
        test_mae = mae(y_test, preds)

        test_maes.append(test_mae)

    return np.mean(test_maes), np.std(test_maes) #return mean and standard dev of 10 runs 


# ---------------------------------------------------------
# MAIN: Run all experiments + save results
# ---------------------------------------------------------
def main():
    print("Running experiments...")

    results = [] #list to save all results 

    for (layer_sizes, activations) in ANN_ARCHITECTURES: #nested loop to try all combinations of ANN and PSO 
        for (swarm, iters, alpha, beta, gamma) in PSO_SETTINGS:

            print(f"\nTesting ANN={layer_sizes}, PSO=[swarm={swarm}, iters={iters}, "
                  f"alpha={alpha}, beta={beta}, gamma={gamma}]")

            mean_mae, std_mae = run_single_configuration(
                layer_sizes, activations,
                swarm, iters, alpha, beta, gamma
            ) #run 10 repetions for this configuration 

            results.append({
                "ANN": str(layer_sizes),
                "Activations": str(activations),
                "Swarm": swarm,
                "Iterations": iters,
                "Alpha": alpha,
                "Beta": beta,
                "Gamma": gamma,
                "Mean_Test_MAE": round(mean_mae, 4),
                "Std_Test_MAE": round(std_mae, 4),
            })

            print(f"Mean Test MAE = {mean_mae:.4f}, Std = {std_mae:.4f}")

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments", "results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print("\n\nAll experiments completed.")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
