import numpy as np
from typing import Callable, Tuple, Optional
from ..base import PopulationBasedOptimizer, OptimizationResult, run_with_timing


class PSO(PopulationBasedOptimizer):
    """Particle Swarm Optimization Algorithm
    
    Attributes:
        w: Inertia weight (controls exploration vs exploitation)
        c1: Cognitive coefficient (personal best influence)
        c2: Social coefficient (global best influence)
        v_max_ratio: Maximum velocity (fraction of search space)
    """
    
    def __init__(self, population_size: int = 30, 
                 w: float = 0.7298, c1: float = 1.49618, c2: float = 1.49618,
                 v_max_ratio: float = 0.2,
                 seed: Optional[int] = None):
        super().__init__(population_size=population_size, seed=seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                initial_population: Optional[np.ndarray] = None,
                **kwargs) -> OptimizationResult:
        """
        Run PSO optimization
        
        Args:
            objective_func: Objective function to optimize
            dim: Problem dimensionality
            bounds: (lower, upper) bounds for each dimension
            max_iter: Maximum number of iterations
            minimize: True for minimization, False for maximization
            initial_population: Optional pre-generated initial population
            
        Returns:
            OptimizationResult with best solution and history
        """
        lower, upper = bounds
        
        # Calculate maximum velocity
        v_max = self.v_max_ratio * (upper - lower)
        
        # Initialize particles and velocities
        if initial_population is not None:
            if len(initial_population) != self.population_size:
                raise ValueError(f"Initial population size {len(initial_population)} does not match population_size {self.population_size}")
            positions = initial_population.copy()
        else:
            positions = self._initialize_population(dim, lower, upper)
            
        velocities = self.rng.uniform(-v_max, v_max, (self.population_size, dim))
        
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
            
            for i in range(self.population_size):
                # Generate random coefficients
                r1 = self.rng.random(dim)
                r2 = self.rng.random(dim)
                
                # Update velocity
                cognitive = self.c1 * r1 * (p_best_positions[i] - positions[i])
                social = self.c2 * r2 * (g_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
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