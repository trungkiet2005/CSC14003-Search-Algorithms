"""Cuckoo Search (CS) Algorithm

Reference: Yang, X. S., & Deb, S. (2009). Cuckoo search via Lévy flights.
"""

import numpy as np


def run_cs(objective_func, dim, bounds, n_nests=25, max_iter=100,
           pa=0.25, beta=1.5, minimize=True, seed=None):
    """Run Cuckoo Search optimization.
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        n_nests: number of nests (solutions)
        max_iter: maximum number of iterations
        pa: probability of discovering alien eggs (0-1)
        beta: parameter for Lévy flights (typically 1.5)
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
    
    # Initialize nests
    nests = np.random.uniform(lower, upper, (n_nests, dim))
    fitness = np.array([objective_func(pos) for pos in nests])
    
    # Initialize best solution
    if minimize:
        best_idx = np.argmin(fitness)
    else:
        best_idx = np.argmax(fitness)
    
    best_nest = nests[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [best_fitness]
    
    for iteration in range(max_iter):
        # Generate new solutions via Lévy flights
        for i in range(n_nests):
            # Lévy flight
            step_size = levy_flight(dim, beta)
            step_size = 0.01 * step_size * (nests[i] - best_nest)
            
            # Generate new solution
            new_nest = nests[i] + step_size
            
            # Boundary handling
            new_nest = np.clip(new_nest, lower, upper)
            
            # Evaluate new solution
            new_fitness = objective_func(new_nest)
            
            # Random nest selection for comparison
            j = np.random.randint(n_nests)
            
            # Replace if better
            if (minimize and new_fitness < fitness[j]) or \
               (not minimize and new_fitness > fitness[j]):
                nests[j] = new_nest
                fitness[j] = new_fitness
        
        # Abandon worst nests (pa fraction)
        n_abandon = int(pa * n_nests)
        if minimize:
            worst_indices = np.argsort(fitness)[-n_abandon:]
        else:
            worst_indices = np.argsort(fitness)[:n_abandon]
        
        for idx in worst_indices:
            # Replace with random solution
            nests[idx] = np.random.uniform(lower, upper, dim)
            fitness[idx] = objective_func(nests[idx])
        
        # Update best solution
        if minimize:
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_nest = nests[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
        else:
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_nest = nests[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
        
        history.append(best_fitness)
    
    return {
        'best_position': best_nest,
        'best_fitness': best_fitness,
        'history': history,
        'nests': nests
    }


def levy_flight(dim, beta=1.5):
    """Generate Lévy flight step using Mantegna's method.
    
    Args:
        dim: dimensionality
        beta: parameter (typically 1.5)
    
    Returns:
        numpy array: Lévy flight step
    """
    # Mantegna's method
    sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    
    step = u / (np.abs(v) ** (1 / beta))
    
    return step

