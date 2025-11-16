# src/pso.py

"""
Particle Swarm Optimisation (PSO) implementation for F20BC PSO–ANN coursework.

Implements Algorithm 39 ("PSO with informants") from:
Sean Luke, *Essentials of Metaheuristics*, 2nd edition.

Key ideas:
- Each particle has a position x and velocity v.
- Each particle tracks its own best position x* (personal best).
- Each particle is influenced by:
    - its personal best x*
    - the best of its informants x+
    - the global best x!
- We treat this as a MAXIMISATION algorithm:
  larger fitness = better. For minimisation (e.g. MAE),
  you can return negative loss from the fitness function.
"""

from __future__ import annotations
from dataclasses import dataclass # MAKES IT EASY TO STORE PARTICLES INFO 
from typing import Callable, Tuple, List #for fitness function

import numpy as np


# Type alias: a fitness function takes a 1D position vector and returns a float.
FitnessFn = Callable[[np.ndarray], float] #takaes a vector (particle position) and returns a number indicating how good that solution is 


@dataclass
class Particle:
    """
    Represents a single PSO particle.
    """
    position: np.ndarray #current solution 
    velocity: np.ndarray #current movement 
    best_pos: np.ndarray #personal best soltuion 
    best_fitness: float #fitness of best position 


class PSO:
    """
    Particle Swarm Optimisation with informants (Algorithm 39).

    Usage:
        pso = PSO(
            dim=dim,
            fitness_fn=my_fitness_fn,
            bounds=(-1.0, 1.0),
            swarm_size=30,
            alpha=0.9,
            beta=0.1,
            gamma=0.1,
            delta=0.0,
            e=1.0,
            n_informants=5,
            rng_seed=0,
        )
        best_pos, best_fit, history = pso.run(max_iter=100)
    """

    def __init__(
        self,
        dim: int, #no of dimensions 
        fitness_fn: FitnessFn, #fucntion to maximise 
        bounds: Tuple[float, float], # min and max values allowed for each dimension 
        swarm_size: int = 30, #no of particles in swarm 
        alpha: float = 0.9,   # line 2: inertia coefficient ; keeps particles moving in same direction 
        beta: float = 0.1,    # line 3: cognitive coefficient ; weight for perosnal best influence 
        gamma: float = 0.1,   # line 4: social (informants) coefficient ; weight for informants influence 
        delta: float = 0.0,   # line 5: global best coefficient (often 0) ; global best infuence 
        e: float = 1.0,       # line 6: step size multiplier 
        n_informants: int = 5, # no of pariticles considered as 'friends'for social learning 
        rng_seed: int = 0, #random seed for reproducibility 
    ) -> None:
        """
        Initialise PSO swarm.

        dim: dimensionality of the search space (length of parameter vector).
        fitness_fn: function mapping a 1D position vector to a scalar fitness.
        bounds: (low, high) for all dimensions; used for initialisation and clamping.
        """
        self.dim = dim
        self.fitness_fn = fitness_fn
        self.bounds = bounds
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.e = e
        self.n_informants = n_informants
        self.rng = np.random.default_rng(rng_seed)

        # Algorithm 39, line 7–9: Initialise swarm
        self.particles: List[Particle] = self._initialise_swarm()

        # Algorithm 39, line 10: Initialise global best (→Best)
        self.global_best_pos: np.ndarray | None = None
        self.global_best_fitness: float = -np.inf

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _initialise_swarm(self) -> List[Particle]:
        """
        Create the initial swarm (Algorithm 39, lines 7–9).
        """
        low, high = self.bounds
        particles: List[Particle] = []

        for _ in range(self.swarm_size):
            # line 9: x_i ← random position, v_i ← random velocity
            position = self.rng.uniform(low, high, size=self.dim)

            # Random initial velocity scaled to half the search range.
            vel_scale = (high - low) * 0.5
            velocity = self.rng.uniform(-vel_scale, vel_scale, size=self.dim)

            particle = Particle(
                position=position,
                velocity=velocity,
                best_pos=position.copy(),  # personal best starts at initial position
                best_fitness=-np.inf,
            )
            particles.append(particle)

        return particles

    def _choose_informants_indices(self) -> List[np.ndarray]:
        """
        For each particle, choose a set of informants (Algorithm 39, used for x+).
        Each particle is always its own informant.
        """
        all_indices = np.arange(self.swarm_size)
        informants_for: List[np.ndarray] = []

        for i in range(self.swarm_size):
            # Choose n_informants - 1 other distinct particles
            others = np.delete(all_indices, i)
            k = max(0, min(self.n_informants - 1, others.size)) #randomly pick other particles 
            chosen = self.rng.choice(others, size=k, replace=False) if k > 0 else np.array([], dtype=int)
            group = np.concatenate([[i], chosen])  # include itself
            informants_for.append(group)

        return informants_for

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self, max_iter: int = 100, verbose: bool = False): #loops over iterations to move particles towards better solutions 
        """
        Run PSO for max_iter iterations.

        Returns:
            best_position: np.ndarray, global best position found (→Best)
            best_fitness: float, best fitness value
            history: list of global best fitness per iteration
        """
        low, high = self.bounds
        history: List[float] = []

        # Algorithm 39, line 11: loop until ideal solution or time is reached
        for iteration in range(max_iter):
            # -------------------------------------------------------------
            # 1) Evaluate swarm and update personal/global bests
            #    (Algorithm 39, lines 12–15)
            # -------------------------------------------------------------
            for p in self.particles:
                # line 13: f_i ← AssessFitness(x_i)
                fitness = self.fitness_fn(p.position)

                # Personal best update (not explicitly numbered, but required)
                if fitness > p.best_fitness:
                    p.best_fitness = fitness
                    p.best_pos = p.position.copy()

                # line 14–15: update global best (→Best)
                if self.global_best_pos is None or fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_pos = p.position.copy()

            history.append(self.global_best_fitness)

            if verbose:
                print(f"[Iter {iteration+1}/{max_iter}] Global best fitness = {self.global_best_fitness:.6f}")

            # -------------------------------------------------------------
            # 2) Velocity update using informants
            #    (Algorithm 39, lines 16–24)
            # -------------------------------------------------------------
            informant_groups = self._choose_informants_indices()

            for idx, p in enumerate(self.particles):
                x = p.position
                v = p.velocity
                x_star = p.best_pos  # line 17: x* (personal best)

                # line 18: x+ ← best of informants
                informants_idx = informant_groups[idx]
                best_informant = None
                best_inf_fit = -np.inf
                for j in informants_idx:
                    pj = self.particles[j]
                    if pj.best_fitness > best_inf_fit:
                        best_inf_fit = pj.best_fitness
                        best_informant = pj
                x_plus = best_informant.best_pos  # best informant position

                # line 19: x! ← global best position
                x_bang = self.global_best_pos

                # line 21–23: sample b ∈ [0, β], c ∈ [0, γ], d ∈ [0, δ] per dimension
                b = self.rng.uniform(0.0, self.beta, size=self.dim)
                c = self.rng.uniform(0.0, self.gamma, size=self.dim)
                d = self.rng.uniform(0.0, self.delta, size=self.dim)

                # line 24:
                # v_i ← α v_i + b (x* - x_i) + c (x+ - x_i) + d (x! - x_i)
                v = (
                    self.alpha * v
                    + b * (x_star - x)
                    + c * (x_plus - x)
                    + d * (x_bang - x)
                )
                p.velocity = v

            # -------------------------------------------------------------
            # 3) Position update (Mutation step)
            #    (Algorithm 39, lines 25–26)
            # -------------------------------------------------------------
            for p in self.particles:
                # line 26: x_i ← x_i + e * v_i
                p.position = p.position + self.e * p.velocity

                # Simple boundary handling: clamp into [low, high].
                # This is a design choice you can extend in "Going further".
                p.position = np.clip(p.position, low, high)

        # Algorithm 39, line 28: return global best
        return self.global_best_pos, self.global_best_fitness, history
