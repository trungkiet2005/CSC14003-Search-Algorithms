"""Simulated Annealing (SA) Algorithm

Probabilistic optimization technique inspired by annealing in metallurgy.
"""

import numpy as np


def run_simulated_annealing(objective_func, dim, bounds, max_iter=1000, 
                           initial_temp=100, cooling_rate=0.95, minimize=True, seed=None):
    """Run Simulated Annealing optimization.
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        max_iter: maximum number of iterations
        initial_temp: initial temperature
        cooling_rate: cooling rate (0-1), typically 0.8-0.99
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
    
    best_position = current_position.copy()
    best_fitness = current_fitness
    
    temperature = initial_temp
    history = [best_fitness]
    
    for iteration in range(max_iter):
        # Generate neighbor solution
        neighbor = current_position + np.random.normal(0, 1, dim) * temperature * 0.01
        neighbor = np.clip(neighbor, lower, upper)
        
        # Evaluate neighbor
        neighbor_fitness = objective_func(neighbor)
        
        # Calculate acceptance probability
        if minimize:
            delta = neighbor_fitness - current_fitness
        else:
            delta = current_fitness - neighbor_fitness
        
        if delta < 0:
            # Accept better solution
            current_position = neighbor
            current_fitness = neighbor_fitness
            
            # Update best
            if (minimize and current_fitness < best_fitness) or \
               (not minimize and current_fitness > best_fitness):
                best_position = current_position.copy()
                best_fitness = current_fitness
        else:
            # Accept worse solution with probability
            acceptance_prob = np.exp(-delta / temperature)
            if np.random.rand() < acceptance_prob:
                current_position = neighbor
                current_fitness = neighbor_fitness
        
        # Cool down
        temperature *= cooling_rate
        
        history.append(best_fitness)
    
    return {
        'best_position': best_position,
        'best_fitness': best_fitness,
        'history': history,
        'final_temp': temperature
    }

