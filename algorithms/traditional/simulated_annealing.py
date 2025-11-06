"""Simulated Annealing (SA) Algorithm

Probabilistic optimization technique inspired by annealing in metallurgy.
"""

import numpy as np


def run_simulated_annealing(objective_func, bounds, dim, max_iter=2500, 
                            initial_temp=1000, final_temp=1e-3, alpha=0.99, 
                            pop_size_equiv=25, seed=None):
    """Run Simulated Annealing algorithm.

    Args:
        objective_func: function to minimize
        bounds: tuple (lower, upper) for each dimension
        dim: dimension of the problem
        max_iter: maximum number of iterations (evaluations)
        initial_temp: starting temperature
        final_temp: final temperature
        alpha: cooling rate
        pop_size_equiv: population size equivalent for history tracking
        seed: random seed

    Returns:
        dict with best fitness, best solution, and history
    """
    rng = np.random.default_rng(seed)
    lower_bound, upper_bound = bounds

    # Initial solution
    current_solution = rng.uniform(lower_bound, upper_bound, dim)
    current_fitness = objective_func(current_solution)

    best_solution = current_solution
    best_fitness = current_fitness

    temp = initial_temp
    history = [best_fitness]

    for i in range(max_iter):
        if temp <= final_temp:
            break

        # Generate neighbor
        neighbor_solution = current_solution + rng.normal(0, 0.5, dim)
        neighbor_solution = np.clip(neighbor_solution, lower_bound, upper_bound)
        neighbor_fitness = objective_func(neighbor_solution)

        # Acceptance probability
        delta = neighbor_fitness - current_fitness
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_solution = neighbor_solution
            current_fitness = neighbor_fitness

            if current_fitness < best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

        # Cool down
        temp *= alpha
        
        # Record history at intervals
        if (i + 1) % pop_size_equiv == 0:
            history.append(best_fitness)

    # Ensure the final best fitness is in the history
    if len(history) < (max_iter // pop_size_equiv) + 1:
        history.append(best_fitness)

    return {
        'best_fitness': best_fitness,
        'best_solution': best_solution,
        'history': history
    }

