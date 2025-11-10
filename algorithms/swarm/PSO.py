"""algorithms/swarm/pso.py - Improved Particle Swarm Optimization

Reference: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
Proceedings of ICNN'95 - International Conference on Neural Networks.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import PopulationBasedOptimizer, OptimizationResult, run_with_timing


class PSO(PopulationBasedOptimizer):
    """Particle Swarm Optimization Algorithm
    
    Attributes:
        n_particles: Number of particles in swarm
        w: Inertia weight (controls exploration vs exploitation)
        c1: Cognitive coefficient (personal best influence)
        c2: Social coefficient (global best influence)
        w_decay: Weight decay rate for adaptive inertia
        v_max: Maximum velocity (fraction of search space)
    """
    
    def __init__(self, n_particles: int = 30, 
                 w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618,
                 w_min: float = 0.4, w_max: float = 0.9,
                 v_max_ratio: float = 0.2,
                 seed: Optional[int] = None):
        super().__init__(population_size=n_particles, seed=seed)
        self.n_particles = n_particles
        self.w_initial = w
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
        self.name = "PSO"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                **kwargs) -> OptimizationResult:
        """
        Run PSO optimization
        
        Args:
            objective_func: Objective function to optimize
            dim: Problem dimensionality
            bounds: (lower, upper) bounds for each dimension
            max_iter: Maximum number of iterations
            minimize: True for minimization, False for maximization
            
        Returns:
            OptimizationResult with best solution and history
        """
        lower, upper = bounds
        
        # Calculate maximum velocity
        v_max = self.v_max_ratio * (upper - lower)
        
        # Initialize particles and velocities
        positions = self._initialize_population(dim, lower, upper)
        velocities = self.rng.uniform(-v_max, v_max, (self.n_particles, dim))
        
        # Evaluate initial fitness
        fitness = self._evaluate_population(positions, objective_func)
        
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
        convergence_iter = None
        
        # Main PSO loop
        for iteration in range(max_iter):
            # Linearly decreasing inertia weight
            w = self.w_max - (self.w_max - self.w_min) * iteration / max_iter
            
            for i in range(self.n_particles):
                # Generate random coefficients
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                
                # Update velocity
                cognitive = self.c1 * r1 * (p_best_positions[i] - positions[i])
                social = self.c2 * r2 * (g_best_position - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                
                # Limit velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Boundary handling
                positions[i] = self._clip_bounds(positions[i], lower, upper)
                
                # Evaluate new position
                fitness[i] = objective_func(positions[i])
                
                # Update personal best
                if (minimize and fitness[i] < p_best_fitness[i]) or \
                   (not minimize and fitness[i] > p_best_fitness[i]):
                    p_best_positions[i] = positions[i].copy()
                    p_best_fitness[i] = fitness[i]
            
            # Update global best
            if minimize:
                best_idx = np.argmin(p_best_fitness)
                if p_best_fitness[best_idx] < g_best_fitness:
                    g_best_position = p_best_positions[best_idx].copy()
                    g_best_fitness = p_best_fitness[best_idx]
            else:
                best_idx = np.argmax(p_best_fitness)
                if p_best_fitness[best_idx] > g_best_fitness:
                    g_best_position = p_best_positions[best_idx].copy()
                    g_best_fitness = p_best_fitness[best_idx]
            
            history.append(g_best_fitness)
            
            # Check convergence
            if convergence_iter is None:
                convergence_iter = self._check_convergence(history)
        
        return OptimizationResult(
            best_position=g_best_position,
            best_fitness=g_best_fitness,
            history=history,
            convergence_iter=convergence_iter,
            final_positions=positions,
            final_velocities=velocities,
            p_best_positions=p_best_positions
        )


def run_pso(objective_func: Callable, dim: int, bounds: Tuple[float, float],
           n_particles: int = 30, max_iter: int = 100,
           w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618,
           w_min: float = 0.4, w_max: float = 0.9, v_max_ratio: float = 0.2,
           minimize: bool = True, seed: Optional[int] = None) -> dict:
    """
    Convenience function to run PSO
    
    Returns dictionary for backward compatibility
    """
    pso = PSO(n_particles=n_particles, w=w, c1=c1, c2=c2, 
              w_min=w_min, w_max=w_max, v_max_ratio=v_max_ratio,
              seed=seed)
    result = pso.optimize(objective_func, dim, bounds, max_iter, minimize)
    return result.to_dict()


# Alias for backward compatibility
def run_pso_old(objective_func, dim, bounds, n_particles=30, max_iter=100,
           w=0.7, c1=1.5, c2=1.5, minimize=True, seed=None):
    """Legacy function signature"""
    return run_pso(objective_func, dim, bounds, n_particles, max_iter,
                  w, c1, c2, minimize, seed)