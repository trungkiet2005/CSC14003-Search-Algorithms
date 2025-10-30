"""Visualization helpers (matplotlib-based) - stubs"""


def plot_history(history, title="Optimization progress"):
    """Plot optimization history (fitness vs iteration).

    `history` expected to be iterable of fitness values.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib required for visualization")

    plt.figure()
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show()
