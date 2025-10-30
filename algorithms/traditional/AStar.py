"""A* Search Algorithm

For pathfinding and discrete optimization problems.
Adapted for TSP with heuristics.
"""

import numpy as np
import heapq


def run_astar_tsp(distance_matrix, max_nodes=10000):
    """Run A* search for TSP.
    
    Uses minimum spanning tree (MST) as heuristic.
    
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
    
    if n_cities > 12:
        print(f"Warning: A* for TSP with {n_cities} cities may be slow. Using limited search.")
    
    # Priority queue: (f_cost, g_cost, current_route, unvisited)
    start_city = 0
    unvisited = set(range(1, n_cities))
    
    initial_h = compute_mst_heuristic(start_city, unvisited, distance_matrix)
    initial_state = (initial_h, 0, [start_city], unvisited)
    
    pq = [initial_state]
    best_route = None
    best_distance = float('inf')
    nodes_explored = 0
    
    while pq and nodes_explored < max_nodes:
        f_cost, g_cost, current_route, unvisited = heapq.heappop(pq)
        nodes_explored += 1
        
        # If all cities visited
        if not unvisited:
            # Complete tour
            final_distance = g_cost + distance_matrix[current_route[-1]][start_city]
            
            if final_distance < best_distance:
                best_distance = final_distance
                best_route = current_route.copy()
            continue
        
        # Pruning: skip if current cost already exceeds best
        if g_cost >= best_distance:
            continue
        
        # Expand to unvisited cities
        current_city = current_route[-1]
        for next_city in unvisited:
            new_g_cost = g_cost + distance_matrix[current_city][next_city]
            new_route = current_route + [next_city]
            new_unvisited = unvisited - {next_city}
            
            # Compute heuristic
            h_cost = compute_mst_heuristic(next_city, new_unvisited, distance_matrix)
            f_cost = new_g_cost + h_cost
            
            heapq.heappush(pq, (f_cost, new_g_cost, new_route, new_unvisited))
    
    return {
        'best_route': best_route,
        'best_distance': best_distance,
        'nodes_explored': nodes_explored
    }


def compute_mst_heuristic(current_city, unvisited, distance_matrix):
    """Compute MST-based heuristic for remaining cities.
    
    Estimates minimum cost to visit all remaining cities.
    """
    if not unvisited:
        return 0
    
    # Include current city in MST computation
    cities = [current_city] + list(unvisited)
    
    if len(cities) == 1:
        return 0
    
    # Prim's algorithm for MST
    visited = {cities[0]}
    mst_cost = 0
    
    while len(visited) < len(cities):
        min_edge = float('inf')
        
        for v in visited:
            for u in cities:
                if u not in visited:
                    edge_cost = distance_matrix[v][u]
                    if edge_cost < min_edge:
                        min_edge = edge_cost
        
        mst_cost += min_edge
        
        # Add city that gives minimum edge (simplified)
        for v in visited:
            for u in cities:
                if u not in visited and distance_matrix[v][u] == min_edge:
                    visited.add(u)
                    break
            if len(visited) == len(cities):
                break
    
    return mst_cost
