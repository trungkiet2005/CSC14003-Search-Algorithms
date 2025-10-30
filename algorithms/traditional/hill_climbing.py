"""Hill Climbing (Steepest Ascent) Algorithm

Classic local search algorithm.
"""

import numpy as np


def run_hill_climbing(objective_func, dim, bounds, max_iter=100, 
                     step_size=0.1, minimize=True, seed=None):
    """Run Hill Climbing optimization (steepest ascent variant).
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        max_iter: maximum number of iterations
        step_size: size of steps to take
        minimize: True for minimization, False for maximization
        seed: random seed for reproducibility
    
    Returns:
        dict with:
            - 'best_position': best solution found
            - 'best_fitness': fitness of best solution
            - 'history': list of best fitness per iteration
    """
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = bounds
    
    # Initialize with random solution
    current_position = np.random.uniform(lower, upper, dim)
    current_fitness = objective_func(current_position)
    
    history = [current_fitness]
    
    for iteration in range(max_iter):
        # Generate neighbors by perturbing each dimension
        improved = False
        
        for d in range(dim):
            # Try positive step
            neighbor = current_position.copy()
            neighbor[d] += step_size
            neighbor = np.clip(neighbor, lower, upper)
            neighbor_fitness = objective_func(neighbor)
            
            if (minimize and neighbor_fitness < current_fitness) or \
               (not minimize and neighbor_fitness > current_fitness):
                current_position = neighbor
                current_fitness = neighbor_fitness
                improved = True
                continue
            
            # Try negative step
            neighbor = current_position.copy()
            neighbor[d] -= step_size
            neighbor = np.clip(neighbor, lower, upper)
            neighbor_fitness = objective_func(neighbor)
            
            if (minimize and neighbor_fitness < current_fitness) or \
               (not minimize and neighbor_fitness > current_fitness):
                current_position = neighbor
                current_fitness = neighbor_fitness
                improved = True
        
        history.append(current_fitness)
        
        # Stop if no improvement (local optimum)
        if not improved:
            # Fill remaining history with current best
            history.extend([current_fitness] * (max_iter - iteration))
            break
    
    return {
        'best_position': current_position,
        'best_fitness': current_fitness,
        'history': history
    }

