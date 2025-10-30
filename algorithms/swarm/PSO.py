"""Particle Swarm Optimization (PSO)

Reference: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
"""

import numpy as np


def run_pso(objective_func, dim, bounds, n_particles=30, max_iter=100, 
            w=0.7, c1=1.5, c2=1.5, minimize=True, seed=None):
    """Run Particle Swarm Optimization.
    
    Args:
        objective_func: function to optimize
        dim: dimensionality of the problem
        bounds: tuple (lower, upper) for search space
        n_particles: number of particles in swarm
        max_iter: maximum number of iterations
        w: inertia weight
        c1: cognitive parameter (personal best)
        c2: social parameter (global best)
        minimize: True for minimization, False for maximization
        seed: random seed for reproducibility
    
    Returns:
        dict with:
            - 'best_position': best solution found
            - 'best_fitness': fitness of best solution
            - 'history': list of best fitness per iteration
            - 'positions': final particle positions
    """
    if seed is not None:
        np.random.seed(seed)
    
    lower, upper = bounds
    
    # Initialize particles
    positions = np.random.uniform(lower, upper, (n_particles, dim))
    velocities = np.random.uniform(-abs(upper - lower) * 0.1, 
                                  abs(upper - lower) * 0.1, 
                                  (n_particles, dim))
    
    # Evaluate initial positions
    fitness = np.array([objective_func(pos) for pos in positions])
    
    # Initialize personal best
    p_best_positions = positions.copy()
    p_best_fitness = fitness.copy()
    
    # Initialize global best
    if minimize:
        g_best_idx = np.argmin(fitness)
    else:
        g_best_idx = np.argmax(fitness)
    
    g_best_position = positions[g_best_idx].copy()
    g_best_fitness = fitness[g_best_idx]
    
    history = [g_best_fitness]
    
    # Main PSO loop
    for iteration in range(max_iter):
        for i in range(n_particles):
            # Update velocity
            r1, r2 = np.random.rand(2)
            
            cognitive = c1 * r1 * (p_best_positions[i] - positions[i])
            social = c2 * r2 * (g_best_position - positions[i])
            
            velocities[i] = w * velocities[i] + cognitive + social
            
            # Update position
            positions[i] = positions[i] + velocities[i]
            
            # Boundary handling
            positions[i] = np.clip(positions[i], lower, upper)
            
            # Evaluate new position
            fitness[i] = objective_func(positions[i])
            
            # Update personal best
            if minimize:
                if fitness[i] < p_best_fitness[i]:
                    p_best_positions[i] = positions[i].copy()
                    p_best_fitness[i] = fitness[i]
            else:
                if fitness[i] > p_best_fitness[i]:
                    p_best_positions[i] = positions[i].copy()
                    p_best_fitness[i] = fitness[i]
        
        # Update global best
        if minimize:
            best_idx = np.argmin(p_best_fitness)
        else:
            best_idx = np.argmax(p_best_fitness)
        
        if minimize:
            if p_best_fitness[best_idx] < g_best_fitness:
                g_best_position = p_best_positions[best_idx].copy()
                g_best_fitness = p_best_fitness[best_idx]
        else:
            if p_best_fitness[best_idx] > g_best_fitness:
                g_best_position = p_best_positions[best_idx].copy()
                g_best_fitness = p_best_fitness[best_idx]
        
        history.append(g_best_fitness)
    
    return {
        'best_position': g_best_position,
        'best_fitness': g_best_fitness,
        'history': history,
        'positions': positions
    }

