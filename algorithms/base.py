"""algorithms/base.py - Base classes for all algorithms"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional, Callable
import time


class OptimizationResult:
    """Container for optimization results"""
    
    def __init__(self, best_position, best_fitness, history, 
                 convergence_iter=None, execution_time=None, **kwargs):
        self.best_position = best_position
        self.best_fitness = best_fitness
        self.history = history
        self.convergence_iter = convergence_iter
        self.execution_time = execution_time
        
        # Store additional algorithm-specific data
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert to dictionary"""
        result = {
            'best_position': self.best_position,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'convergence_iter': self.convergence_iter,
            'execution_time': self.execution_time
        }
        
        # Add any additional attributes
        for key, value in self.__dict__.items():
            if key not in result:
                result[key] = value
        
        return result


class BaseAlgorithm(ABC):
    """Base class for all optimization algorithms"""
    
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.name = self.__class__.__name__
    
    @abstractmethod
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int, **kwargs) -> OptimizationResult:
        """
        Run optimization algorithm
        
        Args:
            objective_func: Function to optimize
            dim: Problem dimensionality
            bounds: Tuple of (lower, upper) bounds
            max_iter: Maximum iterations
            **kwargs: Algorithm-specific parameters
            
        Returns:
            OptimizationResult object
        """
        pass
    
    def _check_convergence(self, history: list, 
                          tolerance: float = 1e-6, 
                          patience: int = 10) -> Optional[int]:
        """
        Check if algorithm has converged
        
        Args:
            history: List of fitness values
            tolerance: Convergence tolerance
            patience: Number of iterations without improvement
            
        Returns:
            Iteration where convergence occurred, or None
        """
        if len(history) < patience:
            return None
        
        recent = history[-patience:]
        if max(recent) - min(recent) < tolerance:
            return len(history) - patience
        
        return None


class ContinuousOptimizer(BaseAlgorithm):
    """Base class for continuous optimization algorithms"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
    
    def _clip_bounds(self, position: np.ndarray, 
                    lower: float, upper: float) -> np.ndarray:
        """Clip position to bounds"""
        return np.clip(position, lower, upper)
    
    def _random_position(self, dim: int, lower: float, upper: float) -> np.ndarray:
        """Generate random position within bounds"""
        return self.rng.uniform(lower, upper, dim)


class DiscreteOptimizer(BaseAlgorithm):
    """Base class for discrete optimization algorithms"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)


class PopulationBasedOptimizer(ContinuousOptimizer):
    """Base class for population-based algorithms"""
    
    def __init__(self, population_size: int, seed: Optional[int] = None):
        super().__init__(seed)
        self.population_size = population_size
    
    def _initialize_population(self, dim: int, 
                              lower: float, upper: float) -> np.ndarray:
        """
        Initialize population.
        """
        return self.rng.uniform(lower, upper, (self.population_size, dim))
    
    def _evaluate_population(self, population: np.ndarray,
                            objective_func: Callable) -> np.ndarray:
        """Evaluate fitness for entire population"""
        return np.array([objective_func(ind) for ind in population])


class LocalSearchOptimizer(ContinuousOptimizer):
    """Base class for local search algorithms"""
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
    
    def _generate_neighbor(self, position: np.ndarray, 
                          step_size: float,
                          lower: float, upper: float) -> np.ndarray:
        """Generate neighbor solution"""
        neighbor = position + self.rng.uniform(-step_size, step_size, len(position))
        return self._clip_bounds(neighbor, lower, upper)


def run_with_timing(func: Callable) -> Callable:
    """Decorator to time algorithm execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        if isinstance(result, OptimizationResult):
            result.execution_time = execution_time
        elif isinstance(result, dict):
            result['execution_time'] = execution_time
        
        return result
    return wrapper


def generate_initial_population(dim: int, bounds: Tuple[float, float], 
                              pop_size: int, seed: Optional[int] = None,
                              avoid_origin_radius: Optional[float] = None) -> np.ndarray:
    """
    Generates a deterministic initial population based on a seed.
    
    Args:
        dim: Problem dimensionality.
        bounds: Tuple of (lower, upper) bounds.
        pop_size: The size of the population to generate.
        seed: The random seed for deterministic generation.
        avoid_origin_radius: If provided, creates an exclusion zone with this radius
                             around the origin.
        
    Returns:
        A NumPy array representing the initial population.
    """
    rng = np.random.default_rng(seed)
    lower, upper = bounds

    # If radius is specified and the bounds span across it, create an exclusion zone
    if avoid_origin_radius is not None and \
       lower < -avoid_origin_radius and upper > avoid_origin_radius:
        
        population = np.zeros((pop_size, dim))
        for i in range(pop_size):
            for j in range(dim):
                if rng.random() < 0.5:
                    # Sample from the lower range: [lower, -radius]
                    population[i, j] = rng.uniform(lower, -avoid_origin_radius)
                else:
                    # Sample from the upper range: [radius, upper]
                    population[i, j] = rng.uniform(avoid_origin_radius, upper)
        return population
    
    # Default behavior
    return rng.uniform(lower, upper, (pop_size, dim))