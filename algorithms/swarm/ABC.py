"""algorithms/swarm/abc.py - Improved Artificial Bee Colony Algorithm

Reference: Karaboga, D., & Basturk, B. (2007). A powerful and efficient 
algorithm for numerical function optimization: artificial bee colony (ABC) algorithm.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import PopulationBasedOptimizer, OptimizationResult, run_with_timing


class ABC(PopulationBasedOptimizer):
    """Artificial Bee Colony Algorithm
    
    Three phases:
    1. Employed bee phase: Each employed bee searches around its food source
    2. Onlooker bee phase: Onlooker bees select food sources based on probability
    3. Scout bee phase: Exhausted food sources are abandoned and replaced
    
    Attributes:
        n_bees: Number of employed bees (total population = 2 * n_bees)
        limit: Abandonment limit for scout bees
        modification_rate: Rate of dimension modification
    """
    
    def __init__(self, n_bees: int = 30, limit: int = None,
                 modification_rate: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(population_size=n_bees, seed=seed)
        self.n_bees = n_bees
        self.limit = limit  # Will be set based on problem if None
        self.modification_rate = modification_rate
        self.name = "ABC"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                initial_population: Optional[np.ndarray] = None,
                **kwargs) -> OptimizationResult:
        """
        Run ABC optimization
        
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
        
        # Set limit if not provided (common: limit = dim * n_bees)
        if self.limit is None:
            limit = dim * self.n_bees
        else:
            limit = self.limit
        
        # Initialize food sources (solutions)
        if initial_population is not None:
            if len(initial_population) != self.n_bees:
                raise ValueError(f"Initial population size {len(initial_population)} does not match n_bees {self.n_bees}")
            food_sources = initial_population.copy()
        else:
            food_sources = self._initialize_population(dim, lower, upper)

        fitness = self._evaluate_population(food_sources, objective_func)
        
        # Convert fitness for probability calculation (higher is better)
        if minimize:
            fitness_values = 1.0 / (1.0 + fitness)
        else:
            fitness_values = fitness.copy()
        
        # Track number of trials for each food source
        trials = np.zeros(self.n_bees, dtype=int)
        
        # Initialize best solution
        if minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)
        
        best_position = food_sources[best_idx].copy()
        best_fitness = fitness[best_idx]
        history = [best_fitness]
        convergence_iter = None
        
        for iteration in range(max_iter):
            # ==================== EMPLOYED BEE PHASE ====================
            for i in range(self.n_bees):
                # Generate new candidate solution
                new_solution = self._generate_neighbor(
                    food_sources, i, dim, lower, upper
                )
                new_fitness = objective_func(new_solution)
                
                # Convert for comparison
                if minimize:
                    new_fitness_val = 1.0 / (1.0 + new_fitness)
                else:
                    new_fitness_val = new_fitness
                
                # Greedy selection
                if (minimize and new_fitness < fitness[i]) or \
                   (not minimize and new_fitness > fitness[i]):
                    food_sources[i] = new_solution
                    fitness[i] = new_fitness
                    fitness_values[i] = new_fitness_val
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            # ==================== ONLOOKER BEE PHASE ====================
            # Calculate selection probabilities
            probabilities = self._calculate_probabilities(fitness_values)
            
            # Onlooker bees select food sources based on probability
            onlooker_count = 0
            t = 0
            while onlooker_count < self.n_bees and t < self.n_bees * 10:
                # Roulette wheel selection
                if self.rng.random() < probabilities[t % self.n_bees]:
                    selected = t % self.n_bees
                    
                    # Generate new candidate solution
                    new_solution = self._generate_neighbor(
                        food_sources, selected, dim, lower, upper
                    )
                    new_fitness = objective_func(new_solution)
                    
                    # Convert for comparison
                    if minimize:
                        new_fitness_val = 1.0 / (1.0 + new_fitness)
                    else:
                        new_fitness_val = new_fitness
                    
                    # Greedy selection
                    if (minimize and new_fitness < fitness[selected]) or \
                       (not minimize and new_fitness > fitness[selected]):
                        food_sources[selected] = new_solution
                        fitness[selected] = new_fitness
                        fitness_values[selected] = new_fitness_val
                        trials[selected] = 0
                    else:
                        trials[selected] += 1
                    
                    onlooker_count += 1
                
                t += 1
            
            # ==================== SCOUT BEE PHASE ====================
            # Find and abandon exhausted food sources
            max_trial_idx = np.argmax(trials)
            if trials[max_trial_idx] >= limit:
                # Scout bee generates new random food source
                food_sources[max_trial_idx] = self._random_position(dim, lower, upper)
                fitness[max_trial_idx] = objective_func(food_sources[max_trial_idx])
                
                if minimize:
                    fitness_values[max_trial_idx] = 1.0 / (1.0 + fitness[max_trial_idx])
                else:
                    fitness_values[max_trial_idx] = fitness[max_trial_idx]
                
                trials[max_trial_idx] = 0
            
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
            
            # Check convergence
            if convergence_iter is None:
                convergence_iter = self._check_convergence(history)
        
        return OptimizationResult(
            best_position=best_position,
            best_fitness=best_fitness,
            history=history,
            convergence_iter=convergence_iter,
            final_food_sources=food_sources,
            final_trials=trials
        )
    
    def _generate_neighbor(self, food_sources: np.ndarray, i: int,
                          dim: int, lower: float, upper: float) -> np.ndarray:
        """Generate neighbor solution for employed/onlooker bee"""
        # Select random neighbor (different from i)
        k = self.rng.integers(0, self.n_bees)
        while k == i:
            k = self.rng.integers(0, self.n_bees)
        
        # Select random dimension(s) to modify
        num_dims_to_modify = max(1, int(self.modification_rate * dim))
        dims_to_modify = self.rng.choice(dim, num_dims_to_modify, replace=False)
        
        # Generate new solution
        new_solution = food_sources[i].copy()
        
        for j in dims_to_modify:
            # phi in [-1, 1]
            phi = self.rng.uniform(-1, 1)
            new_solution[j] = food_sources[i][j] + \
                            phi * (food_sources[i][j] - food_sources[k][j])
        
        # Boundary handling
        new_solution = self._clip_bounds(new_solution, lower, upper)
        
        return new_solution
    
    def _calculate_probabilities(self, fitness_values: np.ndarray) -> np.ndarray:
        """Calculate selection probabilities for onlooker bees"""
        # Ensure all fitness values are positive
        min_fitness = np.min(fitness_values)
        if min_fitness < 0:
            fitness_values = fitness_values - min_fitness + 1e-10
        
        # Add small constant to avoid division by zero
        fitness_values = fitness_values + 1e-10
        
        # Calculate probabilities
        total_fitness = np.sum(fitness_values)
        probabilities = fitness_values / total_fitness
        
        return probabilities


def run_abc(objective_func: Callable, dim: int, bounds: Tuple[float, float],
           n_bees: int = 30, max_iter: int = 100, limit: int = None,
           modification_rate: float = 1.0,
           minimize: bool = True, seed: Optional[int] = None,
           initial_population: Optional[np.ndarray] = None) -> dict:
    """
    Convenience function to run ABC
    
    Returns dictionary for backward compatibility
    """
    abc = ABC(n_bees=n_bees, limit=limit, modification_rate=modification_rate, seed=seed)
    result = abc.optimize(objective_func, dim, bounds, max_iter, minimize,
                          initial_population=initial_population)
    return result.to_dict()