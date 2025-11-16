# src/main.py

import numpy as np

from pso import PSO
from utils import set_global_seed


def dummy_fitness(position: np.ndarray) -> float:
    """
    Example fitness: maximise -sum(x^2), i.e. minimise sum(x^2).
    Optimum at position = 0.
    """
    return -float(np.sum(position ** 2))


def main():
    rng = set_global_seed(42)

    dim = 5
    bounds = (-5.0, 5.0)

    pso = PSO(
        dim=dim,
        fitness_fn=dummy_fitness,
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

    best_pos, best_fit, history = pso.run(max_iter=100, verbose=True)

    print("\n=== PSO dummy test complete ===")
    print("Best fitness:", best_fit)
    print("Best position:", best_pos)


if __name__ == "__main__":
    main()
