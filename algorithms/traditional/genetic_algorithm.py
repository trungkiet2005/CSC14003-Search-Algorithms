"""algorithms/traditional/genetic_algorithm.py - Improved Genetic Algorithm

Reference: Holland, J. H. (1992). Genetic algorithms. Scientific American, 267(1), 66-73.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import PopulationBasedOptimizer, OptimizationResult, run_with_timing


class GeneticAlgorithm(PopulationBasedOptimizer):
    """Genetic Algorithm
    
    Evolutionary algorithm inspired by natural selection and genetics.
    
    Key operations:
    1. Selection: Choose parents based on fitness
    2. Crossover: Combine parents to create offspring
    3. Mutation: Random changes to maintain diversity
    4. Elitism: Preserve best solutions
    
    Attributes:
        pop_size: Population size
        crossover_rate: Probability of crossover (0-1)
        mutation_rate: Probability of mutation (0-1)
        tournament_size: Size of tournament for selection
        elitism_ratio: Fraction of population preserved as elite
    """
    
    def __init__(self, pop_size: int = 50, 
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elitism_ratio: float = 0.1,
                 seed: Optional[int] = None):
        super().__init__(population_size=pop_size, seed=seed)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_ratio = elitism_ratio
        self.name = "GA"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 100, minimize: bool = True,
                initial_population: Optional[np.ndarray] = None,
                **kwargs) -> OptimizationResult:
        """
        Run Genetic Algorithm optimization
        
        Args:
            objective_func: Objective function to optimize
            dim: Problem dimensionality
            bounds: (lower, upper) bounds for each dimension
            max_iter: Maximum number of generations
            minimize: True for minimization, False for maximization
            initial_population: Optional pre-generated initial population
            
        Returns:
            OptimizationResult with best solution and history
        """
        lower, upper = bounds
        
        # Initialize population
        if initial_population is not None:
            if len(initial_population) != self.pop_size:
                raise ValueError(f"Initial population size {len(initial_population)} does not match pop_size {self.pop_size}")
            population = initial_population.copy()
        else:
            population = self._initialize_population(dim, lower, upper)

        fitness = self._evaluate_population(population, objective_func)
        
        # Track best solution
        if minimize:
            best_idx = np.argmin(fitness)
        else:
            best_idx = np.argmax(fitness)
        
        best_position = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        history = [best_fitness]
        convergence_iter = None
        
        # Calculate number of elites
        n_elites = max(1, int(self.elitism_ratio * self.pop_size))
        
        for generation in range(max_iter):
            # ==================== SELECTION ====================
            # Select elite individuals
            if minimize:
                elite_indices = np.argsort(fitness)[:n_elites]
            else:
                elite_indices = np.argsort(fitness)[-n_elites:]
            
            elites = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()
            
            # Select parents for reproduction
            parents = self._tournament_selection(
                population, fitness, self.pop_size - n_elites, minimize
            )
            
            # ==================== CROSSOVER ====================
            offspring = []
            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
                
                if self.rng.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2, dim)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                offspring.extend([child1, child2])
            
            offspring = np.array(offspring[:self.pop_size - n_elites])
            
            # ==================== MUTATION ====================
            offspring = self._mutate(offspring, lower, upper, dim)
            
            # ==================== REPLACEMENT ====================
            # Combine elites and offspring
            population = np.vstack([elites, offspring])
            
            # Evaluate new population
            fitness = self._evaluate_population(population, objective_func)
            
            # Update best solution
            if minimize:
                current_best_idx = np.argmin(fitness)
                if fitness[current_best_idx] < best_fitness:
                    best_position = population[current_best_idx].copy()
                    best_fitness = fitness[current_best_idx]
            else:
                current_best_idx = np.argmax(fitness)
                if fitness[current_best_idx] > best_fitness:
                    best_position = population[current_best_idx].copy()
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
            final_population=population
        )
    
    def _tournament_selection(self, population: np.ndarray, 
                             fitness: np.ndarray,
                             n_select: int, minimize: bool) -> np.ndarray:
        """Select individuals using tournament selection"""
        selected = []
        
        for _ in range(n_select):
            # Random tournament
            tournament_idx = self.rng.choice(
                len(population), self.tournament_size, replace=False
            )
            tournament_fitness = fitness[tournament_idx]
            
            # Select best from tournament
            if minimize:
                winner_idx = tournament_idx[np.argmin(tournament_fitness)]
            else:
                winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray,
                  dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover operation"""
        crossover_type = self.rng.choice(['single_point', 'two_point', 'uniform'])
        
        if crossover_type == 'single_point':
            # Single-point crossover
            point = self.rng.integers(1, dim)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
        
        elif crossover_type == 'two_point':
            # Two-point crossover
            points = sorted(self.rng.choice(range(1, dim), 2, replace=False))
            child1 = np.concatenate([
                parent1[:points[0]], 
                parent2[points[0]:points[1]], 
                parent1[points[1]:]
            ])
            child2 = np.concatenate([
                parent2[:points[0]], 
                parent1[points[0]:points[1]], 
                parent2[points[1]:]
            ])
        
        else:  # uniform crossover
            mask = self.rng.random(dim) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        
        return child1, child2
    
    def _mutate(self, offspring: np.ndarray, lower: float, upper: float,
               dim: int) -> np.ndarray:
        """Perform mutation operation"""
        for i in range(len(offspring)):
            if self.rng.random() < self.mutation_rate:
                mutation_type = self.rng.choice(['gaussian', 'uniform', 'boundary'])
                
                if mutation_type == 'gaussian':
                    # Gaussian mutation
                    mutation_strength = 0.1 * (upper - lower)
                    offspring[i] += self.rng.normal(0, mutation_strength, dim)
                
                elif mutation_type == 'uniform':
                    # Uniform mutation (reset random gene)
                    gene_idx = self.rng.integers(0, dim)
                    offspring[i][gene_idx] = self.rng.uniform(lower, upper)
                
                else:  # boundary mutation
                    # Set random gene to boundary
                    gene_idx = self.rng.integers(0, dim)
                    offspring[i][gene_idx] = self.rng.choice([lower, upper])
                
                # Boundary handling
                offspring[i] = self._clip_bounds(offspring[i], lower, upper)
        
        return offspring


def run_ga(objective_func: Callable, dim: int, bounds: Tuple[float, float],
          pop_size: int = 50, max_iter: int = 100,
          crossover_rate: float = 0.8, mutation_rate: float = 0.1,
          tournament_size: int = 3, elitism_ratio: float = 0.1,
          minimize: bool = True, seed: Optional[int] = None,
          initial_population: Optional[np.ndarray] = None) -> dict:
    """
    Convenience function to run Genetic Algorithm
    
    Returns dictionary for backward compatibility
    """
    ga = GeneticAlgorithm(
        pop_size=pop_size, 
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        tournament_size=tournament_size,
        elitism_ratio=elitism_ratio,
        seed=seed
    )
    result = ga.optimize(objective_func, dim, bounds, max_iter, minimize,
                         initial_population=initial_population)
    return result.to_dict()