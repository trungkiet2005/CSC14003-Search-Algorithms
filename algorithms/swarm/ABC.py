"""Artificial Bee Colony (ABC) Algorithm

Reference: Karaboga, D., & Basturk, B. (2007). 
A powerful and efficient algorithm for numerical function optimization.
"""

import numpy as np


def run_abc(objective_func, dim, bounds, n_bees=30, max_iter=100, 
            limit=50, minimize=True, seed=None):
    """Run Artificial Bee Colony optimization.
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        n_bees: number of employed bees (total bees = 2 * n_bees)
        max_iter: maximum number of iterations
        limit: abandonment limit for scout bees
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
    
    # Initialize food sources (solutions)
    food_sources = np.random.uniform(lower, upper, (n_bees, dim))
    fitness = np.array([objective_func(pos) for pos in food_sources])
    
    # Track number of trials for each food source
    trial = np.zeros(n_bees)
    
    # Initialize best solution
    if minimize:
        best_idx = np.argmin(fitness)
    else:
        best_idx = np.argmax(fitness)
    
    best_position = food_sources[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [best_fitness]
    
    for iteration in range(max_iter):
        # Employed bee phase
        for i in range(n_bees):
            # Generate new candidate solution
            new_solution = employed_bee_phase(food_sources, i, lower, upper, dim)
            new_fitness = objective_func(new_solution)
            
            # Greedy selection
            if (minimize and new_fitness < fitness[i]) or \
               (not minimize and new_fitness > fitness[i]):
                food_sources[i] = new_solution
                fitness[i] = new_fitness
                trial[i] = 0
            else:
                trial[i] += 1
        
        # Calculate probabilities for onlooker bees
        probabilities = calculate_probabilities(fitness, minimize)
        
        # Onlooker bee phase
        for i in range(n_bees):
            # Select food source based on probability
            selected = np.random.choice(n_bees, p=probabilities)
            
            # Generate new candidate solution
            new_solution = employed_bee_phase(food_sources, selected, lower, upper, dim)
            new_fitness = objective_func(new_solution)
            
            # Greedy selection
            if (minimize and new_fitness < fitness[selected]) or \
               (not minimize and new_fitness > fitness[selected]):
                food_sources[selected] = new_solution
                fitness[selected] = new_fitness
                trial[selected] = 0
            else:
                trial[selected] += 1
        
        # Scout bee phase
        for i in range(n_bees):
            if trial[i] > limit:
                # Abandon food source and generate new one
                food_sources[i] = np.random.uniform(lower, upper, dim)
                fitness[i] = objective_func(food_sources[i])
                trial[i] = 0
        
        # Update best solution
        if minimize:
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_position = food_sources[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
        else:
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_position = food_sources[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
        
        history.append(best_fitness)
    
    return {
        'best_position': best_position,
        'best_fitness': best_fitness,
        'history': history,
        'food_sources': food_sources
    }


def employed_bee_phase(food_sources, i, lower, upper, dim):
    """Generate new solution in employed bee phase."""
    n_bees = len(food_sources)
    
    # Select random dimension and neighbor
    k = np.random.randint(n_bees)
    while k == i:
        k = np.random.randint(n_bees)
    
    j = np.random.randint(dim)
    
    # Generate new solution
    phi = np.random.uniform(-1, 1)
    new_solution = food_sources[i].copy()
    new_solution[j] = food_sources[i][j] + phi * (food_sources[i][j] - food_sources[k][j])
    
    # Boundary handling
    new_solution = np.clip(new_solution, lower, upper)
    
    return new_solution


def calculate_probabilities(fitness, minimize):
    """Calculate selection probabilities for onlooker bees."""
    if minimize:
        # For minimization, convert to maximization by inverting
        max_fitness = np.max(fitness)
        adjusted_fitness = max_fitness - fitness + 1e-10
    else:
        adjusted_fitness = fitness - np.min(fitness) + 1e-10
    
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)
    return probabilities

