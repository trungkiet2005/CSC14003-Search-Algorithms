"""Traveling Salesman Problem (TSP)

Provides TSP instances and evaluation functions.
"""

import numpy as np


def euclidean_distance_matrix(cities):
    """Compute Euclidean distance matrix from city coordinates.
    
    Args:
        cities: numpy array of shape (n_cities, 2) with (x, y) coordinates
    
    Returns:
        numpy array of shape (n_cities, n_cities) with distances
    """
    n = len(cities)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(cities[i] - cities[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
    
    return dist_matrix


def total_distance(route, distance_matrix):
    """Compute total distance for a TSP route.
    
    Args:
        route: list or array of city indices representing the tour
        distance_matrix: 2D numpy array of distances
    
    Returns:
        float: total tour distance
    """
    route = np.asarray(route, dtype=int)
    distance = 0.0
    
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        distance += distance_matrix[from_city][to_city]
    
    return distance


def generate_random_tsp(n_cities, seed=None, area_size=100):
    """Generate a random TSP instance.
    
    Args:
        n_cities: number of cities
        seed: random seed for reproducibility
        area_size: size of the square area for city placement
    
    Returns:
        tuple: (cities, distance_matrix)
            cities: numpy array of shape (n_cities, 2)
            distance_matrix: numpy array of shape (n_cities, n_cities)
    """
    if seed is not None:
        np.random.seed(seed)
    
    cities = np.random.rand(n_cities, 2) * area_size
    distance_matrix = euclidean_distance_matrix(cities)
    
    return cities, distance_matrix


def create_tsp_problem(n_cities=20, seed=42):
    """Create a TSP problem instance for testing.
    
    Args:
        n_cities: number of cities
        seed: random seed
    
    Returns:
        dict with 'cities', 'distance_matrix', and 'objective' function
    """
    cities, dist_matrix = generate_random_tsp(n_cities)
    
    def objective(route):
        """Objective function for TSP (tour length - to minimize)."""
        return total_distance(route, dist_matrix)
    
    return {
        'cities': cities,
        'distance_matrix': dist_matrix,
        'objective': objective,
        'n_cities': n_cities
    }
