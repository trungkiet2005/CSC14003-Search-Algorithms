import numpy as np
from typing import Callable, Tuple, Optional
from ..base import PopulationBasedOptimizer, OptimizationResult, run_with_timing
import math


class CS(PopulationBasedOptimizer):
    """Cuckoo Search Algorithm
    
    Based on the brood parasitism of cuckoo species combined with Lévy flights.
    
    Key concepts:
    1. Each cuckoo lays one egg (solution) at a time
    2. Best nests (solutions) carry over to next generation
    3. Host bird can discover alien egg with probability pa
    
    Attributes:
        pa: Discovery probability (fraction of worst nests abandoned)
        beta: Lévy distribution parameter (typically 1.5)
        step_size_factor: Step size scaling factor
    """
    
    def __init__(self, population_size: int = 25, pa: float = 0.25,
                 beta: float = 1.5, step_size_factor: float = 0.01,
                 seed: Optional[int] = None):
        super().__init__(population_size=population_size, seed=seed)
        self.pa = pa
        self.beta = beta
        self.step_size_factor = step_size_factor
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                initial_population: Optional[np.ndarray] = None,
                **kwargs) -> OptimizationResult:
        """
        Run Cuckoo Search optimization
        
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
        
        # Initialize nests
        if initial_population is not None:
            if len(initial_population) != self.population_size:
                raise ValueError(f"Initial population size {len(initial_population)} does not match population_size {self.population_size}")
            nests = initial_population.copy()
        else:
            nests = self._initialize_population(dim, lower, upper)

        fitness = self._evaluate_population(nests, objective_func)
        
        # Initialize best solution
        if minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)
        
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]
        history = [best_fitness]
        convergence_iter = None
        
        # Calculate search space scale
        scale = upper - lower
        
        for iteration in range(max_iter):
            # ==================== LÉVY FLIGHTS ====================
            # Generate new solutions via Lévy flights
            for i in range(self.population_size):
                # Generate Lévy flight step
                step = self._levy_flight(dim)
                
                # Scale step size
                step_size = self.step_size_factor * step * scale
                
                # Direction towards best solution
                direction = best_nest - nests[i]
                
                # Generate new solution
                new_nest = nests[i] + step_size * direction + \
                          self.rng.normal(0, 0.01 * scale, dim)
                
                # Boundary handling
                new_nest = self._clip_bounds(new_nest, lower, upper)
                
                # Evaluate new solution
                new_fitness = objective_func(new_nest)
                
                # Random walk: compare with random nest
                j = self.rng.integers(0, self.population_size)
                
                # Replace if better
                if (minimize and new_fitness < fitness[j]) or \
                   (not minimize and new_fitness > fitness[j]):
                    nests[j] = new_nest
                    fitness[j] = new_fitness
            
            # ==================== ABANDON WORST NESTS ====================
            # Abandon a fraction pa of worst nests
            n_abandon = max(1, int(self.pa * self.population_size))
            
            if minimize:
                worst_indices = np.argsort(fitness)[-n_abandon:]
            else:
                worst_indices = np.argsort(fitness)[:n_abandon]
            
            # Replace worst nests with new random solutions
            # Use biased random walk around best solutions
            for idx in worst_indices:
                if self.rng.random() < self.pa:
                    # Generate new solution using random walk
                    # Select two random nests
                    k1, k2 = self.rng.choice(self.population_size, 2, replace=False)
                    
                    # Random walk
                    step_size = self.rng.random() * (nests[k1] - nests[k2])
                    new_nest = nests[idx] + step_size
                    
                    # Boundary handling
                    new_nest = self._clip_bounds(new_nest, lower, upper)
                    
                    nests[idx] = new_nest
                    fitness[idx] = objective_func(new_nest)
            
            # ==================== UPDATE BEST ====================
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
            
            # Check convergence
            if convergence_iter is None:
                convergence_iter = self._check_convergence(history)
        
        return OptimizationResult(
            best_position=best_nest,
            best_fitness=best_fitness,
            history=history,
            convergence_iter=convergence_iter,
            final_nests=nests
        )
    
    def _levy_flight(self, dim: int) -> np.ndarray:
        """
        Generate Lévy flight step using Mantegna's method
        
        Lévy distribution: L(s) ~ s^(-1-β) for large s
        
        Args:
            dim: Dimensionality of the step
            
        Returns:
            Lévy flight step vector
        """
        beta = self.beta
        
        # Calculate sigma using Mantegna's method
        numerator = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
        denominator = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
        sigma_u = (numerator / denominator) ** (1 / beta)
        
        # Generate step
        u = self.rng.normal(0, sigma_u, dim)
        v = self.rng.normal(0, 1, dim)
        
        # Lévy step
        step = u / (np.abs(v) ** (1 / beta))
        
        return step