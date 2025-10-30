"""Knapsack Problem (0/1 Knapsack)

Binary optimization problem for discrete algorithms.
"""

import numpy as np


def knapsack_value(solution, values, weights, capacity):
    """Compute total value of a knapsack solution.
    
    Args:
        solution: binary array (1 = item included, 0 = not included)
        values: array of item values
        weights: array of item weights
        capacity: maximum weight capacity
    
    Returns:
        float: total value if valid, 0 if over capacity (penalty)
    """
    solution = np.asarray(solution, dtype=int)
    total_weight = np.sum(solution * weights)
    total_value = np.sum(solution * values)
    
    # Penalize if over capacity
    if total_weight > capacity:
        return 0.0
    
    return total_value


def generate_knapsack_instance(n_items, seed=None):
    """Generate a random knapsack instance.
    
    Args:
        n_items: number of items
        seed: random seed for reproducibility
    
    Returns:
        dict with 'values', 'weights', 'capacity'
    """
    if seed is not None:
        np.random.seed(seed)
    
    values = np.random.randint(1, 100, size=n_items)
    weights = np.random.randint(1, 50, size=n_items)
    capacity = int(np.sum(weights) * 0.5)  # 50% of total weight
    
    return {
        'values': values,
        'weights': weights,
        'capacity': capacity,
        'n_items': n_items
    }


def create_knapsack_problem(n_items=30, seed=42):
    """Create a knapsack problem instance for testing.
    
    Args:
        n_items: number of items
        seed: random seed
    
    Returns:
        dict with problem data and objective function
    """
    instance = generate_knapsack_instance(n_items, seed=seed)
    
    def objective(solution):
        """Objective function for knapsack (value - to maximize).
        
        Note: Return negative value for minimization algorithms.
        """
        return knapsack_value(solution, 
                            instance['values'], 
                            instance['weights'], 
                            instance['capacity'])
    
    return {
        **instance,
        'objective': objective
    }
