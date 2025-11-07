"""problems/tsp.py - Enhanced Traveling Salesman Problem Implementation"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class TSPInstance:
    """TSP problem instance"""
    n_cities: int
    cities: np.ndarray  # (n_cities, 2) coordinates
    distance_matrix: np.ndarray  # (n_cities, n_cities) distances
    optimal_distance: Optional[float] = None
    name: str = "Random TSP"


class TSPProblem:
    """Traveling Salesman Problem"""
    
    def __init__(self, distance_matrix: np.ndarray, cities: Optional[np.ndarray] = None):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.cities = cities
    
    def evaluate(self, route: list) -> float:
        """Calculate total tour distance"""
        return calculate_route_distance(route, self.distance_matrix)
    
    def __call__(self, route: list) -> float:
        """Make problem callable"""
        return self.evaluate(route)
    
    def is_valid_route(self, route: list) -> bool:
        """Check if route is valid (visits each city exactly once)"""
        return (len(route) == self.n_cities and 
                len(set(route)) == self.n_cities and
                all(0 <= city < self.n_cities for city in route))
    
    def two_opt_improvement(self, route: list) -> list:
        """Apply 2-opt local search improvement"""
        improved = True
        best_route = route.copy()
        best_distance = self.evaluate(best_route)
        
        while improved:
            improved = False
            for i in range(1, self.n_cities - 1):
                for j in range(i + 1, self.n_cities):
                    # Try reversing segment [i:j]
                    new_route = best_route.copy()
                    new_route[i:j] = reversed(new_route[i:j])
                    new_distance = self.evaluate(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                
                if improved:
                    break
        
        return best_route


def euclidean_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix from city coordinates
    
    Args:
        cities: numpy array of shape (n_cities, 2) with (x, y) coordinates
    
    Returns:
        Distance matrix of shape (n_cities, n_cities)
    """
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    return dist_matrix


def calculate_route_distance(route: list, distance_matrix: np.ndarray) -> float:
    """
    Calculate total distance of a TSP tour
    
    Args:
        route: List of city indices representing the tour
        distance_matrix: 2D numpy array of distances
    
    Returns:
        Total tour distance
    """
    route = np.asarray(route, dtype=int)
    distance = 0.0
    
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        distance += distance_matrix[from_city][to_city]
    
    return distance


def generate_random_tsp(n_cities: int, seed: Optional[int] = None, 
                       area_size: float = 100.0) -> TSPInstance:
    """
    Generate a random TSP instance
    
    Args:
        n_cities: Number of cities
        seed: Random seed for reproducibility
        area_size: Size of the square area for city placement
    
    Returns:
        TSPInstance object
    """
    rng = np.random.default_rng(seed)
    
    # Generate random city coordinates
    cities = rng.uniform(0, area_size, (n_cities, 2))
    
    # Calculate distance matrix
    distance_matrix = euclidean_distance_matrix(cities)
    
    return TSPInstance(
        n_cities=n_cities,
        cities=cities,
        distance_matrix=distance_matrix,
        name=f"Random_{n_cities}cities"
    )


def generate_clustered_tsp(n_cities: int, n_clusters: int = 3,
                          seed: Optional[int] = None,
                          area_size: float = 100.0) -> TSPInstance:
    """
    Generate a clustered TSP instance (more realistic)
    
    Args:
        n_cities: Number of cities
        n_clusters: Number of clusters
        seed: Random seed
        area_size: Size of area
    
    Returns:
        TSPInstance object
    """
    rng = np.random.default_rng(seed)
    
    cities_per_cluster = n_cities // n_clusters
    cities = []
    
    # Generate cluster centers
    cluster_centers = rng.uniform(area_size * 0.2, area_size * 0.8, (n_clusters, 2))
    
    # Generate cities around clusters
    for i in range(n_clusters):
        center = cluster_centers[i]
        cluster_radius = area_size * 0.15
        
        # Generate cities in this cluster
        for _ in range(cities_per_cluster):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, cluster_radius)
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            cities.append([x, y])
    
    # Add remaining cities if n_cities not divisible by n_clusters
    remaining = n_cities - len(cities)
    for _ in range(remaining):
        cities.append(rng.uniform(0, area_size, 2))
    
    cities = np.array(cities)
    distance_matrix = euclidean_distance_matrix(cities)
    
    return TSPInstance(
        n_cities=n_cities,
        cities=cities,
        distance_matrix=distance_matrix,
        name=f"Clustered_{n_cities}cities_{n_clusters}clusters"
    )


def generate_grid_tsp(grid_size: int, seed: Optional[int] = None) -> TSPInstance:
    """
    Generate TSP instance on a grid
    
    Args:
        grid_size: Size of grid (total cities = grid_size^2)
        seed: Random seed
    
    Returns:
        TSPInstance object
    """
    n_cities = grid_size * grid_size
    cities = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            cities.append([i, j])
    
    cities = np.array(cities, dtype=float)
    
    # Add small random noise
    if seed is not None:
        rng = np.random.default_rng(seed)
        noise = rng.uniform(-0.1, 0.1, cities.shape)
        cities += noise
    
    distance_matrix = euclidean_distance_matrix(cities)
    
    return TSPInstance(
        n_cities=n_cities,
        cities=cities,
        distance_matrix=distance_matrix,
        name=f"Grid_{grid_size}x{grid_size}"
    )


def create_tsp_problem(n_cities: int = 20, seed: int = 42,
                      problem_type: str = 'random') -> Dict[str, Any]:
    """
    Create a TSP problem instance for testing
    
    Args:
        n_cities: Number of cities
        seed: Random seed
        problem_type: 'random', 'clustered', or 'grid'
    
    Returns:
        Dictionary with problem information
    """
    if problem_type == 'clustered':
        instance = generate_clustered_tsp(n_cities, seed=seed)
    elif problem_type == 'grid':
        grid_size = int(np.sqrt(n_cities))
        instance = generate_grid_tsp(grid_size, seed=seed)
    else:  # random
        instance = generate_random_tsp(n_cities, seed=seed)
    
    problem = TSPProblem(instance.distance_matrix, instance.cities)
    
    return {
        'cities': instance.cities,
        'distance_matrix': instance.distance_matrix,
        'n_cities': instance.n_cities,
        'objective': problem,
        'name': instance.name,
        'problem_type': problem_type
    }


def nearest_neighbor_heuristic(distance_matrix: np.ndarray, 
                               start_city: int = 0) -> Tuple[list, float]:
    """
    Construct tour using nearest neighbor heuristic
    
    Args:
        distance_matrix: Distance matrix
        start_city: Starting city
    
    Returns:
        Tuple of (route, distance)
    """
    n_cities = len(distance_matrix)
    route = [start_city]
    unvisited = set(range(n_cities)) - {start_city}
    current = start_city
    
    while unvisited:
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda city: distance_matrix[current][city])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    
    distance = calculate_route_distance(route, distance_matrix)
    return route, distance


def get_tsp_lower_bound(distance_matrix: np.ndarray) -> float:
    """
    Calculate lower bound for TSP using minimum spanning tree
    
    Args:
        distance_matrix: Distance matrix
    
    Returns:
        Lower bound estimate
    """
    n = len(distance_matrix)
    
    # Simple lower bound: sum of two smallest edges for each city
    lower_bound = 0.0
    for i in range(n):
        edges = sorted([distance_matrix[i][j] for j in range(n) if i != j])
        lower_bound += sum(edges[:2])
    
    return lower_bound / 2  # Each edge counted twice