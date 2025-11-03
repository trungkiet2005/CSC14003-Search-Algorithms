"""Firefly Algorithm (FA)

Reference: Yang, X. S. (2009). Firefly algorithms for multimodal optimization.
"""

import numpy as np


def run_fa(objective_func, dim, bounds, n_fireflies=25, max_iter=100,
           alpha=0.5, beta0=1.0, gamma=1.0, minimize=True, seed=None):
    """Run Firefly Algorithm optimization.
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        n_fireflies: number of fireflies
        max_iter: maximum number of iterations
        alpha: randomization parameter
        beta0: attractiveness at distance 0
        gamma: light absorption coefficient
        minimize: True for minimization, False for maximization
        seed: random seed for reproducibility
    
    Returns:
        dict with:
            - 'best_position': best solution found
            - 'best_fitness': fitness of best solution
            - 'history': list of best fitness per iteration
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds
    
    # Initialize firefly positions
    fireflies = rng.uniform(lower, upper, (n_fireflies, dim))
    intensity = np.array([objective_func(pos) for pos in fireflies])
    
    # For minimization, lower values have higher intensity
    if minimize:
        light_intensity = -intensity
    else:
        light_intensity = intensity
    
    # Initialize best solution
    best_idx = np.argmax(light_intensity)
    best_position = fireflies[best_idx].copy()
    best_fitness = intensity[best_idx]
    history = [best_fitness]
    
    for iteration in range(max_iter):
        # Update alpha (reduce randomness over time)
        alpha_t = alpha * (0.95 ** iteration)
        
        # Create a copy of fireflies for this iteration's calculations
        new_fireflies = fireflies.copy()

        # Move fireflies
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                # If firefly j is brighter than firefly i
                if light_intensity[j] > light_intensity[i]:
                    # Calculate distance based on original positions
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    
                    # Calculate attractiveness
                    beta = beta0 * np.exp(-gamma * r ** 2)
                    
                    # Move firefly i towards j (update the new_fireflies array)
                    random_move = alpha_t * (rng.random(dim) - 0.5)
                    new_fireflies[i] += beta * (fireflies[j] - fireflies[i]) + random_move

        # Apply boundary constraints to the new positions
        new_fireflies = np.clip(new_fireflies, lower, upper)
        fireflies = new_fireflies
        
        # Evaluate new positions
        for i in range(n_fireflies):
            intensity[i] = objective_func(fireflies[i])
            light_intensity[i] = -intensity[i] if minimize else intensity[i]
        
        # Update best solution
        current_best_idx = np.argmax(light_intensity)
        if minimize:
            if intensity[current_best_idx] < best_fitness:
                best_position = fireflies[current_best_idx].copy()
                best_fitness = intensity[current_best_idx]
        else:
            if intensity[current_best_idx] > best_fitness:
                best_position = fireflies[current_best_idx].copy()
                best_fitness = intensity[current_best_idx]
        
        history.append(best_fitness)
    
    return {
        'best_position': best_position,
        'best_fitness': best_fitness,
        'history': history,
        'fireflies': fireflies
    }

