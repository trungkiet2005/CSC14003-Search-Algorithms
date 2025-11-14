"""algorithms/traditional/hill_climbing.py - Improved Hill Climbing Algorithm

Reference: Russell, S., & Norvig, P. (2010). Artificial Intelligence: A Modern Approach.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import LocalSearchOptimizer, OptimizationResult, run_with_timing


class HillClimbing(LocalSearchOptimizer):
    """Hill Climbing (Steepest Ascent) Algorithm
    
    Local search algorithm that iteratively moves to the best neighboring solution.
    
    Attributes:
        step_size: Initial step size for generating neighbors
        adaptive_step: Whether to adapt step size over time
        random_restart: Number of random restarts to escape local optima
    """
    
    def __init__(self, step_size: float = 0.1,
                 adaptive_step: bool = True,
                 random_restart: int = 5,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)
        self.initial_step_size = step_size
        self.adaptive_step = adaptive_step
        self.random_restart = random_restart
        self.name = "HC"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                **kwargs) -> OptimizationResult:
        """
        Run Hill Climbing optimization with random restarts
        
        Args:
            objective_func: Objective function to optimize
            dim: Problem dimensionality
            bounds: (lower, upper) bounds for each dimension
            max_iter: Maximum number of iterations per restart
            minimize: True for minimization, False for maximization
            
        Returns:
            OptimizationResult with best solution and history
        """
        lower, upper = bounds
        scale = upper - lower
        
        # Global best across all restarts
        global_best_position = None
        global_best_fitness = float('inf') if minimize else float('-inf')
        global_history = []
        convergence_iter = None
        
        # Perform multiple random restarts
        for restart in range(self.random_restart):
            # Initialize with random solution
            current_position = self._random_position(dim, lower, upper)
            current_fitness = objective_func(current_position)
            
            # Track best in this restart
            restart_best_position = current_position.copy()
            restart_best_fitness = current_fitness
            
            # Adaptive step size
            step_size = self.initial_step_size * scale
            
            no_improvement_count = 0
            
            for iteration in range(max_iter):
                improved = False
                
                # Try all dimensions (steepest ascent)
                best_neighbor = None
                best_neighbor_fitness = current_fitness
                
                for d in range(dim):
                    # Try positive and negative steps in each dimension
                    for direction in [1, -1]:
                        # Generate neighbor
                        neighbor = current_position.copy()
                        neighbor[d] += direction * step_size
                        
                        # Boundary handling
                        neighbor = self._clip_bounds(neighbor, lower, upper)
                        
                        # Evaluate neighbor
                        neighbor_fitness = objective_func(neighbor)
                        
                        # Check if this is the best neighbor so far
                        if (minimize and neighbor_fitness < best_neighbor_fitness) or \
                           (not minimize and neighbor_fitness > best_neighbor_fitness):
                            best_neighbor = neighbor
                            best_neighbor_fitness = neighbor_fitness
                            improved = True
                
                # Move to best neighbor if found
                if improved:
                    current_position = best_neighbor
                    current_fitness = best_neighbor_fitness
                    no_improvement_count = 0
                    
                    # Update restart best
                    if (minimize and current_fitness < restart_best_fitness) or \
                       (not minimize and current_fitness > restart_best_fitness):
                        restart_best_position = current_position.copy()
                        restart_best_fitness = current_fitness
                else:
                    no_improvement_count += 1
                
                # Adaptive step size reduction
                if self.adaptive_step and no_improvement_count > 0:
                    step_size *= 0.95  # Reduce step size
                    
                    # If step size too small, stop this restart
                    if step_size < 1e-6 * scale:
                        break
                
                # Record history
                global_history.append(restart_best_fitness)
            
            # Update global best
            if (minimize and restart_best_fitness < global_best_fitness) or \
               (not minimize and restart_best_fitness > global_best_fitness):
                global_best_position = restart_best_position.copy()
                global_best_fitness = restart_best_fitness
        
        # Check convergence
        if global_history:
            convergence_iter = self._check_convergence(global_history)
        
        return OptimizationResult(
            best_position=global_best_position,
            best_fitness=global_best_fitness,
            history=global_history,
            convergence_iter=convergence_iter,
            n_restarts=self.random_restart
        )


class StochasticHillClimbing(LocalSearchOptimizer):
    """Stochastic Hill Climbing
    
    Variant that randomly selects among improving neighbors
    rather than always choosing the best.
    """
    
    def __init__(self, step_size: float = 0.1,
                 n_neighbors: int = 20,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)
        self.step_size = step_size
        self.n_neighbors = n_neighbors
        self.name = "Stochastic-HC"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                **kwargs) -> OptimizationResult:
        """Run Stochastic Hill Climbing"""
        lower, upper = bounds
        scale = upper - lower
        
        # Initialize
        current_position = self._random_position(dim, lower, upper)
        current_fitness = objective_func(current_position)
        
        best_position = current_position.copy()
        best_fitness = current_fitness
        history = [best_fitness]
        
        step_size = self.step_size * scale
        
        for iteration in range(max_iter):
            # Generate random neighbors
            improved = False
            
            for _ in range(self.n_neighbors):
                # Random perturbation
                neighbor = current_position + self.rng.normal(0, step_size, dim)
                neighbor = self._clip_bounds(neighbor, lower, upper)
                
                # Evaluate
                neighbor_fitness = objective_func(neighbor)
                
                # Accept if better
                if (minimize and neighbor_fitness < current_fitness) or \
                   (not minimize and neighbor_fitness > current_fitness):
                    current_position = neighbor
                    current_fitness = neighbor_fitness
                    improved = True
                    break  # Accept first improvement (first-choice)
            
            # Update best
            if (minimize and current_fitness < best_fitness) or \
               (not minimize and current_fitness > best_fitness):
                best_position = current_position.copy()
                best_fitness = current_fitness
            
            history.append(best_fitness)
            
            # Reduce step size if no improvement
            if not improved:
                step_size *= 0.95
        
        return OptimizationResult(
            best_position=best_position,
            best_fitness=best_fitness,
            history=history
        )


# Convenience functions
def run_hill_climbing(objective_func: Callable, dim: int, 
                     bounds: Tuple[float, float],
                     max_iter: int = 100, step_size: float = 0.1,
                     adaptive_step: bool = True,
                     random_restart: int = 5,
                     minimize: bool = True, 
                     seed: Optional[int] = None,
                     **kwargs) -> dict:
    """
    Convenience function to run Hill Climbing
    
    Returns dictionary for backward compatibility
    """
    hc = HillClimbing(
        step_size=step_size,
        adaptive_step=adaptive_step,
        random_restart=random_restart,
        seed=seed
    )
    result = hc.optimize(objective_func, dim, bounds, max_iter, minimize)
    return result.to_dict()


def run_stochastic_hill_climbing(objective_func: Callable, dim: int,
                                 bounds: Tuple[float, float],
                                 max_iter: int = 100,
                                 step_size: float = 0.1,
                                 n_neighbors: int = 20,
                                 minimize: bool = True,
                                 seed: Optional[int] = None,
                                 **kwargs) -> dict:
    """Run Stochastic Hill Climbing"""
    shc = StochasticHillClimbing(
        step_size=step_size,
        n_neighbors=n_neighbors,
        seed=seed
    )
    result = shc.optimize(objective_func, dim, bounds, max_iter, minimize)
    return result.to_dict()
