"""Sphere function benchmark problem"""

import math


def sphere(x):
    """Compute the Sphere function for a vector x (list/tuple).

    f(x) = sum(x_i^2)
    Global minimum at x=0 with f=0.
    """
    return sum((xi ** 2) for xi in x)
