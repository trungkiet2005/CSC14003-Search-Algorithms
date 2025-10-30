"""Depth-First Search (DFS)

For discrete optimization problems - explores search space depth-first.
Adapted for TSP and discrete optimization.
"""

import numpy as np


def run_dfs_tsp(distance_matrix, max_nodes=5000):
    """Run DFS for TSP (limited due to exponential complexity).
    
    Note: DFS for TSP is only practical for very small instances (n < 10).
    
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
        print(f"Warning: DFS for TSP with {n_cities} cities is impractical. Using limited search.")
    
    best_solution = {
        'route': None,
        'distance': float('inf'),
        'nodes_explored': 0
    }
    
    # Start DFS from city 0
    visited = [False] * n_cities
    current_route = []
    
    dfs_helper(0, visited, current_route, 0, distance_matrix, best_solution, max_nodes)
    
    return {
        'best_route': best_solution['route'],
        'best_distance': best_solution['distance'],
        'nodes_explored': best_solution['nodes_explored']
    }


def dfs_helper(current_city, visited, current_route, current_distance, 
               distance_matrix, best_solution, max_nodes):
    """Recursive DFS helper function."""
    
    if best_solution['nodes_explored'] >= max_nodes:
        return
    
    # Mark current city as visited
    visited[current_city] = True
    current_route.append(current_city)
    best_solution['nodes_explored'] += 1
    
    n_cities = len(distance_matrix)
    
    # If all cities visited
    if len(current_route) == n_cities:
        # Complete the tour by returning to start
        total_distance = current_distance + distance_matrix[current_city][current_route[0]]
        
        if total_distance < best_solution['distance']:
            best_solution['distance'] = total_distance
            best_solution['route'] = current_route.copy()
    else:
        # Try visiting unvisited cities
        for next_city in range(n_cities):
            if not visited[next_city]:
                new_distance = current_distance + distance_matrix[current_city][next_city]
                
                # Pruning: skip if current path is already worse than best
                if new_distance < best_solution['distance']:
                    dfs_helper(next_city, visited, current_route, new_distance,
                             distance_matrix, best_solution, max_nodes)
    
    # Backtrack
    visited[current_city] = False
    current_route.pop()
