"""Benchmark helper functions"""

import time


def timeit(func, *args, repeat=1, **kwargs):
    """Run func and measure average execution time over `repeat` runs.

    Returns (result, avg_time_seconds).
    """
    total = 0.0
    result = None
    for _ in range(repeat):
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        total += (t1 - t0)
    return result, total / repeat
