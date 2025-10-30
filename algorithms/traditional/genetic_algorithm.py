"""Genetic Algorithm (GA)

Evolutionary algorithm inspired by natural selection.
"""

import numpy as np


def run_ga(objective_func, dim, bounds, pop_size=50, max_iter=100,
          crossover_rate=0.8, mutation_rate=0.1, minimize=True, seed=None):
    """Run Genetic Algorithm optimization.
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        pop_size: population size
        max_iter: maximum number of generations
        crossover_rate: probability of crossover (0-1)
        mutation_rate: probability of mutation (0-1)
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
    
    # Initialize population
    population = np.random.uniform(lower, upper, (pop_size, dim))
    fitness = np.array([objective_func(ind) for ind in population])
    
    # Track best solution
    if minimize:
        best_idx = np.argmin(fitness)
    else:
        best_idx = np.argmax(fitness)
    
    best_position = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    history = [best_fitness]
    
    for generation in range(max_iter):
        # Selection (tournament selection)
        parents = tournament_selection(population, fitness, minimize, pop_size)
        
        # Crossover
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < pop_size else parents[0]
            
            if np.random.rand() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            offspring.extend([child1, child2])
        
        offspring = np.array(offspring[:pop_size])
        
        # Mutation
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:
                offspring[i] = mutate(offspring[i], lower, upper)
        
        # Boundary handling
        offspring = np.clip(offspring, lower, upper)
        
        # Evaluate offspring
        offspring_fitness = np.array([objective_func(ind) for ind in offspring])
        
        # Replacement (elitism: keep best individual)
        population = offspring
        fitness = offspring_fitness
        
        # Ensure best is preserved
        if minimize:
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_position = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
            else:
                # Replace worst with best from previous generation
                worst_idx = np.argmax(fitness)
                population[worst_idx] = best_position
                fitness[worst_idx] = best_fitness
        else:
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_position = population[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]
            else:
                # Replace worst with best from previous generation
                worst_idx = np.argmin(fitness)
                population[worst_idx] = best_position
                fitness[worst_idx] = best_fitness
        
        history.append(best_fitness)
    
    return {
        'best_position': best_position,
        'best_fitness': best_fitness,
        'history': history,
        'population': population
    }


def tournament_selection(population, fitness, minimize, n_select, tournament_size=3):
    """Select individuals using tournament selection."""
    selected = []
    pop_size = len(population)
    
    for _ in range(n_select):
        # Random tournament
        tournament_idx = np.random.choice(pop_size, tournament_size, replace=False)
        tournament_fitness = fitness[tournament_idx]
        
        # Select best from tournament
        if minimize:
            winner_idx = tournament_idx[np.argmin(tournament_fitness)]
        else:
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
        
        selected.append(population[winner_idx].copy())
    
    return np.array(selected)


def crossover(parent1, parent2):
    """Perform single-point crossover."""
    point = np.random.randint(1, len(parent1))
    
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    
    return child1, child2


def mutate(individual, lower, upper):
    """Perform Gaussian mutation."""
    mutation_strength = 0.1 * (upper - lower)
    mutated = individual + np.random.normal(0, mutation_strength, len(individual))
    return mutated

