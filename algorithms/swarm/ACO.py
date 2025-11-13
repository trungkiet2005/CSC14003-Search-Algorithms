"""algorithms/swarm/aco.py - Improved Ant Colony Optimization for TSP

Reference: Dorigo, M., & Stützle, T. (2006). Ant Colony Optimization.
"""

import numpy as np
from typing import Optional
from ..base import DiscreteOptimizer, OptimizationResult, run_with_timing

class ACO(DiscreteOptimizer):
    """Ant Colony System (ACS) bases on ACO metaheuristic for TSP — Dorigo & Gambardella 1996"""

    def __init__(self, n_ants: int = 20, alpha: float = 1.0, beta: float = 2.0,
                 evaporation_rate: float = 0.1, phi: float = 0.1, q0: float = 0.9,
                 seed: Optional[int] = None):
        super().__init__(seed)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate              # evaporation for offline update
        self.phi = phi              # local update rate
        self.q0 = q0                # pseudorandom proportional parameter
        self.name = "ACO"

    def optimize(self, objective_func: callable, dim: int, bounds: tuple, max_iter: int, **kwargs) -> OptimizationResult:
        pass

    @run_with_timing
    def optimize_tsp(self, distance_matrix: np.ndarray,
                     max_iter: int = 100) -> OptimizationResult:
        n_cities = len(distance_matrix)
        off_diag = distance_matrix[np.triu_indices(n_cities, k=1)]
        mean_off = np.mean(off_diag[off_diag > 0]) if np.any(off_diag > 0) else 1.0
        tau0 = 1.0 / (n_cities * mean_off)
        pheromone = np.ones((n_cities, n_cities)) * tau0

        with np.errstate(divide='ignore', invalid='ignore'):
            heuristic = 1.0 / distance_matrix
            heuristic[~np.isfinite(heuristic)] = 0

        best_route = None
        best_distance = float('inf')
        history = []

        for iteration in range(max_iter):
            all_routes = []
            all_distances = []

            for ant in range(self.n_ants):
                route = self._construct_route(pheromone, heuristic, tau0)
                distance = self._calculate_route_distance(route, distance_matrix)

                all_routes.append(route)
                all_distances.append(distance)

                if distance < best_distance:
                    best_distance = distance
                    best_route = route.copy()

            history.append(best_distance)

            # Offline pheromone update (only best ant)
            pheromone *= (1 - self.evaporation_rate)
            for i in range(len(best_route)):
                a, b = best_route[i], best_route[(i + 1) % len(best_route)]
                pheromone[a, b] += self.evaporation_rate * (1.0 / best_distance)
                pheromone[b, a] = pheromone[a, b]

        return OptimizationResult(
            best_position=best_route,
            best_fitness=best_distance,
            history=history,
            best_route=best_route,
            best_distance=best_distance,
            final_pheromone=pheromone
        )

    def _construct_route(self, pheromone, heuristic, tau0):
        n_cities = pheromone.shape[0]
        route = []
        visited = np.zeros(n_cities, dtype=bool)
        current_city = self.rng.integers(0, n_cities)
        route.append(current_city)
        visited[current_city] = True

        for _ in range(n_cities - 1):
            unvisited = np.where(~visited)[0]
            tau = pheromone[current_city, unvisited] ** self.alpha
            eta = heuristic[current_city, unvisited] ** self.beta
            desirability = tau * eta

            # Pseudorandom proportional rule
            if self.rng.random() < self.q0:
                # Exploitation: choose the best path
                next_city = unvisited[np.argmax(desirability)]
            else:
                # Exploration: choose based on probability
                sum_des = np.sum(desirability)
                if sum_des > 0:
                    probs = desirability / sum_des
                    next_city = self.rng.choice(unvisited, p=probs)
                else:
                    # Fallback: if all paths have zero desirability, choose randomly
                    next_city = self.rng.choice(unvisited)

            # Local pheromone update
            new_val = (1 - self.phi) * pheromone[current_city, next_city] + self.phi * tau0
            pheromone[current_city, next_city] = new_val
            pheromone[next_city, current_city] = new_val

            route.append(next_city)
            visited[next_city] = True
            current_city = next_city

        return route

    def _calculate_route_distance(self, route, distance_matrix):
        distance = 0.0
        for i in range(len(route)):
            a, b = route[i], route[(i + 1) % len(route)]
            distance += distance_matrix[a, b]
        return distance


def run_aco(distance_matrix: np.ndarray, 
            n_ants: int = 20, 
            max_iter: int = 100,
            alpha: float = 1.0, 
            beta: float = 2.0, 
            evaporation_rate: float = 0.1, 
            phi: float = 0.1, 
            q0: float = 0.9,
            seed: Optional[int] = None) -> dict:
    """
    Run Ant Colony System (ACS) for TSP — backward-compatible name.
    """
    acs = ACO(
        n_ants=n_ants, alpha=alpha, beta=beta,
        evaporation_rate=evaporation_rate, phi=phi, q0=q0, seed=seed
    )
    result = acs.optimize_tsp(distance_matrix, max_iter)
    return result.to_dict()