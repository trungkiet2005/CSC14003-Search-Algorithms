"""TSP problem stub (representation and distance helper)

Add full TSP instance parsing and solver interfaces here.
"""

import math


def total_distance(route, distance_matrix):
    """Compute total distance for a route (list of city indices).

    distance_matrix: 2D list or dict providing distances.
    """
    d = 0.0
    for i in range(len(route)):
        a = route[i]
        b = route[(i + 1) % len(route)]
        d += distance_matrix[a][b]
    return d
