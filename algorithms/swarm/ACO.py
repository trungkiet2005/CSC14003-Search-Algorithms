"""algorithms/swarm/aco.py - Improved Ant Colony Optimization for TSP

Reference: Dorigo, M., & Stützle, T. (2004). Ant Colony Optimization.
"""

import numpy as np
from typing import Optional, Tuple
from ..base import DiscreteOptimizer, OptimizationResult, run_with_timing


class ACO(DiscreteOptimizer):
    """Ant Colony Optimization for TSP
    
    Attributes:
        n_ants: Number of ants
        alpha: Pheromone importance factor
        beta: Heuristic importance factor (visibility)
        evaporation_rate: Pheromone evaporation rate (0-1)
        Q: Pheromone deposit factor
        elitist_weight: Weight for elitist ant strategy
    """
    
    def __init__(self, n_ants: int = 20, alpha: float = 1.0, beta: float = 2.0,
                 evaporation_rate: float = 0.5, Q: float = 100,
                 elitist_weight: float = 2.0, seed: Optional[int] = None):
        super().__init__(seed)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.elitist_weight = elitist_weight
        self.name = "ACO"
    
    def optimize(self, objective_func: callable, dim: int, bounds: tuple, max_iter: int, **kwargs) -> OptimizationResult:
        pass

    @run_with_timing
    def optimize_tsp(self, distance_matrix: np.ndarray, 
                     max_iter: int = 100) -> OptimizationResult:
        """
        Run ACO for TSP
        
        Args:
            distance_matrix: 2D numpy array of distances between cities
            max_iter: Maximum number of iterations
            
        Returns:
            OptimizationResult with best tour and history
        """
        n_cities = len(distance_matrix)
        
        # Initialize pheromone matrix
        initial_pheromone = 1.0 / (n_cities * np.mean(distance_matrix))
        pheromone = np.ones((n_cities, n_cities)) * initial_pheromone
        
        # Heuristic information (visibility = 1/distance)
        with np.errstate(divide='ignore', invalid='ignore'):
            heuristic = 1.0 / distance_matrix
            heuristic[~np.isfinite(heuristic)] = 0
        
        # Track best solution
        best_route = None
        best_distance = float('inf')
        history = []
        convergence_iter = None
        stagnation_counter = 0
        
        for iteration in range(max_iter):
            all_routes = []
            all_distances = []
            
            # Each ant constructs a solution
            for ant in range(self.n_ants):
                route = self._construct_route(pheromone, heuristic, n_cities)
                distance = self._calculate_route_distance(route, distance_matrix)
                
                all_routes.append(route)
                all_distances.append(distance)
                
                # Update best solution
                if distance < best_distance:
                    best_distance = distance
                    best_route = route.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            
            history.append(best_distance)
            
            # Update pheromone with elitist strategy
            pheromone = self._update_pheromone(
                pheromone, all_routes, all_distances, 
                best_route, best_distance
            )
            
            # Check convergence
            if convergence_iter is None:
                convergence_iter = self._check_convergence(history)
            
            # Restart if stagnation detected
            if stagnation_counter > 50:
                pheromone = np.ones((n_cities, n_cities)) * initial_pheromone
                stagnation_counter = 0
        
        return OptimizationResult(
            best_position=best_route,
            best_fitness=best_distance,
            history=history,
            convergence_iter=convergence_iter,
            best_route=best_route,
            best_distance=best_distance,
            final_pheromone=pheromone
        )
    
    def _construct_route(self, pheromone: np.ndarray, 
                        heuristic: np.ndarray, n_cities: int) -> list:
        """Construct a route for one ant using probabilistic selection"""
        route = []
        visited = np.zeros(n_cities, dtype=bool)
        
        # Start from random city
        current_city = self.rng.integers(0, n_cities)
        route.append(current_city)
        visited[current_city] = True
        
        # Construct route by selecting next cities
        for _ in range(n_cities - 1):
            # Calculate probabilities for unvisited cities
            unvisited = np.where(~visited)[0]
            
            # Pheromone and heuristic factors
            tau = pheromone[current_city, unvisited] ** self.alpha
            eta = heuristic[current_city, unvisited] ** self.beta
            
            probabilities = tau * eta
            
            # Handle case where all probabilities are zero
            if probabilities.sum() == 0:
                probabilities = np.ones(len(unvisited))
            
            # Normalize probabilities
            probabilities = probabilities / probabilities.sum()
            
            # Select next city
            next_city_idx = self.rng.choice(len(unvisited), p=probabilities)
            next_city = unvisited[next_city_idx]
            
            route.append(next_city)
            visited[next_city] = True
            current_city = next_city
        
        return route
    
    def _calculate_route_distance(self, route: list, 
                                  distance_matrix: np.ndarray) -> float:
        """Calculate total distance of a route"""
        distance = 0.0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            distance += distance_matrix[from_city, to_city]
        return distance
    
    def _update_pheromone(self, pheromone: np.ndarray, 
                         routes: list, distances: list,
                         best_route: list, best_distance: float) -> np.ndarray:
        """Update pheromone matrix with elitist strategy"""
        n_cities = len(pheromone)
        
        # Evaporation
        pheromone *= (1 - self.evaporation_rate)
        
        # Deposit pheromone from all ants
        for route, distance in zip(routes, distances):
            deposit = self.Q / distance
            for i in range(len(route)):
                from_city = route[i]
                to_city = route[(i + 1) % len(route)]
                pheromone[from_city, to_city] += deposit
                pheromone[to_city, from_city] += deposit
        
        # Elitist ant strategy - reinforce best solution
        if best_route is not None:
            elitist_deposit = self.elitist_weight * self.Q / best_distance
            for i in range(len(best_route)):
                from_city = best_route[i]
                to_city = best_route[(i + 1) % len(best_route)]
                pheromone[from_city, to_city] += elitist_deposit
                pheromone[to_city, from_city] += elitist_deposit
        
        return pheromone


def run_aco(distance_matrix: np.ndarray, n_ants: int = 20, max_iter: int = 100,
           alpha: float = 1.0, beta: float = 2.0, evaporation_rate: float = 0.5,
           Q: float = 100, seed: Optional[int] = None) -> dict:
    """
    Convenience function to run ACO for TSP
    
    Returns dictionary for backward compatibility
    """
    aco = ACO(n_ants=n_ants, alpha=alpha, beta=beta, 
              evaporation_rate=evaporation_rate, Q=Q, seed=seed)
    result = aco.optimize_tsp(distance_matrix, max_iter)
    return result.to_dict()