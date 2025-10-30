"""Ant Colony Optimization (ACO) for TSP

Reference: Dorigo, M., & Stützle, T. (2004). Ant colony optimization.
Optimized for discrete combinatorial problems like TSP.
"""

import numpy as np


def run_aco(distance_matrix, n_ants=20, max_iter=100, alpha=1.0, beta=2.0,
            evaporation_rate=0.5, Q=100, seed=None):
    """Run Ant Colony Optimization for TSP.
    
    Args:
        distance_matrix: 2D numpy array of distances between cities
        n_ants: number of ants
        max_iter: maximum number of iterations
        alpha: pheromone importance factor
        beta: heuristic importance factor (visibility)
        evaporation_rate: pheromone evaporation rate (0-1)
        Q: pheromone deposit factor
        seed: random seed for reproducibility
    
    Returns:
        dict with:
            - 'best_route': best tour found
            - 'best_distance': length of best tour
            - 'history': list of best distance per iteration
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_cities = len(distance_matrix)
    
    # Initialize pheromone matrix
    pheromone = np.ones((n_cities, n_cities)) / n_cities
    
    # Heuristic information (visibility = 1/distance)
    with np.errstate(divide='ignore'):
        heuristic = 1.0 / distance_matrix
    heuristic[heuristic == np.inf] = 0
    
    best_route = None
    best_distance = float('inf')
    history = []
    
    for iteration in range(max_iter):
        all_routes = []
        all_distances = []
        
        # Each ant constructs a solution
        for ant in range(n_ants):
            route = construct_route(pheromone, heuristic, alpha, beta, n_cities)
            distance = calculate_route_distance(route, distance_matrix)
            
            all_routes.append(route)
            all_distances.append(distance)
            
            # Update best solution
            if distance < best_distance:
                best_distance = distance
                best_route = route.copy()
        
        history.append(best_distance)
        
        # Update pheromone
        pheromone = update_pheromone(pheromone, all_routes, all_distances,
                                     evaporation_rate, Q)
    
    return {
        'best_route': best_route,
        'best_distance': best_distance,
        'history': history
    }


def construct_route(pheromone, heuristic, alpha, beta, n_cities):
    """Construct a route for one ant using probabilistic selection.
    
    Args:
        pheromone: pheromone matrix
        heuristic: heuristic information matrix
        alpha: pheromone importance
        beta: heuristic importance
        n_cities: number of cities
    
    Returns:
        list: route (permutation of city indices)
    """
    route = []
    visited = set()
    
    # Start from random city
    current_city = np.random.randint(n_cities)
    route.append(current_city)
    visited.add(current_city)
    
    # Construct route by selecting next cities
    while len(route) < n_cities:
        probabilities = []
        unvisited = []
        
        for city in range(n_cities):
            if city not in visited:
                tau = pheromone[current_city][city] ** alpha
                eta = heuristic[current_city][city] ** beta
                probabilities.append(tau * eta)
                unvisited.append(city)
        
        # Normalize probabilities
        probabilities = np.array(probabilities)
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()
        else:
            probabilities = np.ones(len(unvisited)) / len(unvisited)
        
        # Select next city
        next_city = np.random.choice(unvisited, p=probabilities)
        route.append(next_city)
        visited.add(next_city)
        current_city = next_city
    
    return route


def calculate_route_distance(route, distance_matrix):
    """Calculate total distance of a route."""
    distance = 0.0
    for i in range(len(route)):
        from_city = route[i]
        to_city = route[(i + 1) % len(route)]
        distance += distance_matrix[from_city][to_city]
    return distance


def update_pheromone(pheromone, routes, distances, evaporation_rate, Q):
    """Update pheromone matrix based on ant solutions.
    
    Args:
        pheromone: current pheromone matrix
        routes: list of routes from all ants
        distances: list of distances for all routes
        evaporation_rate: evaporation rate
        Q: pheromone deposit factor
    
    Returns:
        numpy array: updated pheromone matrix
    """
    n_cities = len(pheromone)
    
    # Evaporation
    pheromone = (1 - evaporation_rate) * pheromone
    
    # Deposit pheromone
    for route, distance in zip(routes, distances):
        deposit = Q / distance
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i + 1) % len(route)]
            pheromone[from_city][to_city] += deposit
            pheromone[to_city][from_city] += deposit  # Symmetric
    
    return pheromone

