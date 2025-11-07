"""algorithms/traditional/simulated_annealing.py - Improved Simulated Annealing

Reference: Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). 
Optimization by simulated annealing. Science, 220(4598), 671-680.
"""

import numpy as np
from typing import Callable, Tuple, Optional
from ..base import LocalSearchOptimizer, OptimizationResult, run_with_timing


class SimulatedAnnealing(LocalSearchOptimizer):
    """Simulated Annealing Algorithm
    
    Probabilistic optimization inspired by annealing in metallurgy.
    Accepts worse solutions with probability that decreases over time.
    
    Attributes:
        initial_temp: Starting temperature
        final_temp: Final temperature (stopping criterion)
        alpha: Cooling rate (exponential: T = T * alpha)
        cooling_schedule: Type of cooling ('exponential', 'linear', 'logarithmic')
        neighbor_std: Standard deviation for neighbor generation
    """
    
    def __init__(self, initial_temp: float = 1000, 
                 final_temp: float = 1e-3,
                 alpha: float = 0.99,
                 cooling_schedule: str = 'exponential',
                 neighbor_std: float = 0.5,
                 seed: Optional[int] = None):
        super().__init__(seed=seed)
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.cooling_schedule = cooling_schedule
        self.neighbor_std = neighbor_std
        self.name = "SA"
    
    @run_with_timing
    def optimize(self, objective_func: Callable, 
                dim: int, bounds: Tuple[float, float],
                max_iter: int = 2500, minimize: bool = True,
                **kwargs) -> OptimizationResult:
        """
        Run Simulated Annealing optimization
        
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
        scale = upper - lower
        
        # Initial solution
        current_solution = self._random_position(dim, lower, upper)
        current_fitness = objective_func(current_solution)
        
        # Best solution
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        # Temperature
        temp = self.initial_temp
        
        # History tracking
        history = [best_fitness]
        convergence_iter = None
        acceptance_history = []
        
        iteration = 0
        while temp > self.final_temp and iteration < max_iter:
            # Generate neighbor solution
            neighbor_solution = self._generate_neighbor_solution(
                current_solution, scale, lower, upper
            )
            neighbor_fitness = objective_func(neighbor_solution)
            
            # Calculate energy difference
            if minimize:
                delta_e = neighbor_fitness - current_fitness
            else:
                delta_e = current_fitness - neighbor_fitness
            
            # Acceptance criterion
            if delta_e < 0:
                # Accept better solution
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                acceptance_history.append(1)
            else:
                # Accept worse solution with probability
                acceptance_prob = np.exp(-delta_e / temp)
                if self.rng.random() < acceptance_prob:
                    current_solution = neighbor_solution
                    current_fitness = neighbor_fitness
                    acceptance_history.append(1)
                else:
                    acceptance_history.append(0)
            
            # Update best solution
            if (minimize and current_fitness < best_fitness) or \
               (not minimize and current_fitness > best_fitness):
                best_solution = current_solution.copy()
                best_fitness = current_fitness
            
            # Cool down
            temp = self._cool_down(temp, iteration, max_iter)
            
            # Record history periodically
            if iteration % 25 == 0:
                history.append(best_fitness)
            
            iteration += 1
        
        # Ensure final best is recorded
        if history[-1] != best_fitness:
            history.append(best_fitness)
        
        # Check convergence
        convergence_iter = self._check_convergence(history)
        
        return OptimizationResult(
            best_position=best_solution,
            best_fitness=best_fitness,
            history=history,
            convergence_iter=convergence_iter,
            final_temperature=temp,
            acceptance_rate=np.mean(acceptance_history) if acceptance_history else 0
        )
    
    def _generate_neighbor_solution(self, current: np.ndarray, scale: float,
                                   lower: float, upper: float) -> np.ndarray:
        """Generate neighbor solution using Gaussian perturbation"""
        perturbation = self.rng.normal(0, self.neighbor_std * scale, len(current))
        neighbor = current + perturbation
        return self._clip_bounds(neighbor, lower, upper)
    
    def _cool_down(self, temp: float, iteration: int, max_iter: int) -> float:
        """Apply cooling schedule"""
        if self.cooling_schedule == 'exponential':
            return temp * self.alpha
        
        elif self.cooling_schedule == 'linear':
            return self.initial_temp - (self.initial_temp - self.final_temp) * \
                   iteration / max_iter
        
        elif self.cooling_schedule == 'logarithmic':
            return self.initial_temp / (1 + np.log(1 + iteration))
        
        else:
            return temp * self.alpha


class SimulatedAnnealingTSP:
    """Simulated Annealing for TSP (discrete optimization)"""
    
    def __init__(self, initial_temp: float = 1000,
                 final_temp: float = 1e-3,
                 alpha: float = 0.995,
                 seed: Optional[int] = None):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.name = "SA-TSP"
    
    @run_with_timing
    def optimize_tsp(self, distance_matrix: np.ndarray,
                    max_iter: int = 20000) -> OptimizationResult:
        """
        Run SA for TSP
        
        Args:
            distance_matrix: 2D numpy array of distances
            max_iter: Maximum iterations
            
        Returns:
            OptimizationResult with best tour
        """
        n_cities = len(distance_matrix)
        
        # Initial solution (random route)
        current_route = list(self.rng.permutation(n_cities))
        current_distance = self._calculate_route_distance(
            current_route, distance_matrix
        )
        
        # Best solution
        best_route = current_route.copy()
        best_distance = current_distance
        
        # Temperature
        temp = self.initial_temp
        
        history = [best_distance]
        
        iteration = 0
        while temp > self.final_temp and iteration < max_iter:
            # Generate neighbor using 2-opt swap
            neighbor_route = self._two_opt_swap(current_route)
            neighbor_distance = self._calculate_route_distance(
                neighbor_route, distance_matrix
            )
            
            # Calculate delta
            delta_e = neighbor_distance - current_distance
            
            # Acceptance criterion
            if delta_e < 0 or self.rng.random() < np.exp(-delta_e / temp):
                current_route = neighbor_route
                current_distance = neighbor_distance
                
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            # Cool down
            temp *= self.alpha
            
            # Record history
            if iteration % 100 == 0:
                history.append(best_distance)
            
            iteration += 1
        
        history.append(best_distance)
        
        return OptimizationResult(
            best_position=best_route,
            best_fitness=best_distance,
            history=history,
            best_route=best_route,
            best_distance=best_distance
        )
    
    def _two_opt_swap(self, route: list) -> list:
        """Perform 2-opt swap on route"""
        new_route = route.copy()
        i, j = sorted(self.rng.choice(len(route), 2, replace=False))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route
    
    def _calculate_route_distance(self, route: list, 
                                  distance_matrix: np.ndarray) -> float:
        """Calculate total distance of route"""
        distance = 0.0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            distance += distance_matrix[from_city, to_city]
        return distance


# Convenience functions
def run_simulated_annealing(objective_func: Callable, dim: int, 
                           bounds: Tuple[float, float],
                           max_iter: int = 2500, initial_temp: float = 1000,
                           final_temp: float = 1e-3, alpha: float = 0.99,
                           minimize: bool = True, seed: Optional[int] = None,
                           **kwargs) -> dict:
    """Run SA for continuous optimization"""
    sa = SimulatedAnnealing(
        initial_temp=initial_temp,
        final_temp=final_temp,
        alpha=alpha,
        seed=seed
    )
    result = sa.optimize(objective_func, dim, bounds, max_iter, minimize)
    return result.to_dict()


def run_simulated_annealing_tsp(distance_matrix: np.ndarray,
                               max_iter: int = 20000,
                               initial_temp: float = 1000,
                               final_temp: float = 1e-3,
                               alpha: float = 0.995,
                               seed: Optional[int] = None) -> dict:
    """Run SA for TSP"""
    sa_tsp = SimulatedAnnealingTSP(
        initial_temp=initial_temp,
        final_temp=final_temp,
        alpha=alpha,
        seed=seed
    )
    result = sa_tsp.optimize_tsp(distance_matrix, max_iter)
    return result.to_dict()