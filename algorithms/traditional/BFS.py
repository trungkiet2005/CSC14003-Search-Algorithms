"""Breadth-First Search (BFS)

For discrete optimization problems - explores search space level by level.
Adapted for TSP and discrete optimization.
"""

import numpy as np
from collections import deque


def run_bfs_tsp(distance_matrix, max_nodes=5000):
    """Run BFS for TSP (limited due to exponential complexity).
    
    Note: BFS for TSP is only practical for very small instances (n < 10).
    This implementation uses a limited search.
    
    Args:
        distance_matrix: 2D numpy array of distances
        max_nodes: maximum nodes to explore
    
    Returns:
        dict with:
            - 'best_route': best tour found
            - 'best_distance': length of best tour
            - 'nodes_explored': number of nodes explored
    """
    n_cities = len(distance_matrix)
    
    if n_cities > 10:
        print(f"Warning: BFS for TSP with {n_cities} cities is impractical. Using greedy heuristic.")
        return run_bfs_greedy_tsp(distance_matrix)
    
    # BFS queue: each state is (current_route, unvisited_set)
    start_city = 0
    initial_state = ([start_city], set(range(1, n_cities)))
    queue = deque([initial_state])
    
    best_route = None
    best_distance = float('inf')
    nodes_explored = 0
    
    while queue and nodes_explored < max_nodes:
        current_route, unvisited = queue.popleft()
        nodes_explored += 1
        
        # If all cities visited, complete the tour
        if not unvisited:
            # Return to start
            complete_route = current_route + [start_city]
            distance = calculate_tour_distance(complete_route, distance_matrix)
            
            if distance < best_distance:
                best_distance = distance
                best_route = current_route
            continue
        
        # Expand to unvisited cities
        current_city = current_route[-1]
        for next_city in unvisited:
            new_route = current_route + [next_city]
            new_unvisited = unvisited - {next_city}
            queue.append((new_route, new_unvisited))
    
    return {
        'best_route': best_route,
        'best_distance': best_distance,
        'nodes_explored': nodes_explored
    }


def run_bfs_greedy_tsp(distance_matrix):
    """Greedy nearest neighbor heuristic for TSP (BFS-like)."""
    n_cities = len(distance_matrix)
    
    best_route = None
    best_distance = float('inf')
    
    # Try starting from each city
    for start in range(n_cities):
        route = [start]
        unvisited = set(range(n_cities)) - {start}
        
        current = start
        while unvisited:
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda city: distance_matrix[current][city])
            route.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Calculate distance
        distance = calculate_tour_distance(route + [start], distance_matrix)
        
        if distance < best_distance:
            best_distance = distance
            best_route = route
    
    return {
        'best_route': best_route,
        'best_distance': best_distance,
        'nodes_explored': n_cities
    }


def calculate_tour_distance(route, distance_matrix):
    """Calculate total distance of a tour."""
    distance = 0.0
    for i in range(len(route) - 1):
        distance += distance_matrix[route[i]][route[i + 1]]
    return distance
