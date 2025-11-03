"""Visualization helpers for optimization algorithms"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_history(history, title="Optimization Progress", save_path=None):
    """Plot optimization history (fitness vs iteration).
    
    Args:
        history: iterable of fitness values
        title: plot title
        save_path: path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2)
    plt.title(title, fontsize=14)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence_comparison(histories_dict, title="Convergence Comparison", 
                               save_path=None, log_scale=False, xlabel="Iteration"):
    """Compare convergence of multiple algorithms.
    
    Args:
        histories_dict: dict of {algorithm_name: history}
        title: plot title
        save_path: path to save figure
        log_scale: use log scale for y-axis
        xlabel: label for the x-axis (e.g., "Iteration", "Nodes Expanded")
    """
    plt.figure(figsize=(12, 7))
    
    for name, history in histories_dict.items():
        plt.plot(history, label=name, linewidth=2, alpha=0.8)
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Fitness" + (" (log scale)" if log_scale else ""), fontsize=12)
    
    if log_scale:
        plt.yscale('log')
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_3d_surface(func, bounds, resolution=50, title="3D Function Surface", 
                   save_path=None, best_point=None):
    """Plot 3D surface of a 2D objective function.
    
    Args:
        func: objective function (takes 2D array)
        bounds: tuple (lower, upper) for both dimensions
        resolution: number of points per dimension
        title: plot title
        save_path: path to save figure
        best_point: optional (x, y) point to mark on surface
    """
    lower, upper = bounds
    
    # Create mesh grid
    x = np.linspace(lower, upper, resolution)
    y = np.linspace(lower, upper, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    # Mark best point if provided
    if best_point is not None:
        z_best = func(best_point)
        ax.scatter([best_point[0]], [best_point[1]], [z_best], 
                  color='red', s=100, marker='*', label='Best Solution')
        ax.legend()
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('f(X, Y)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_contour(func, bounds, resolution=100, title="Contour Plot", 
                save_path=None, best_point=None, particle_positions=None):
    """Plot contour of a 2D objective function.
    
    Args:
        func: objective function
        bounds: tuple (lower, upper)
        resolution: number of points per dimension
        title: plot title
        save_path: path to save figure
        best_point: optional (x, y) point to mark
        particle_positions: optional array of particle positions to plot
    """
    lower, upper = bounds
    
    # Create mesh grid
    x = np.linspace(lower, upper, resolution)
    y = np.linspace(lower, upper, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = func([X[i, j], Y[i, j]])
    
    plt.figure(figsize=(10, 8))
    
    # Contour plot
    contour = plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.3)
    plt.colorbar(contour, label='Fitness')
    
    # Plot particles if provided
    if particle_positions is not None:
        positions = np.array(particle_positions)
        if positions.shape[1] >= 2:
            plt.scatter(positions[:, 0], positions[:, 1], 
                       c='blue', s=30, alpha=0.6, label='Particles')
    
    # Mark best point
    if best_point is not None:
        plt.scatter([best_point[0]], [best_point[1]], 
                   color='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2,
                   label='Best Solution', zorder=5)
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_boxplot_comparison(data_dict, title="Algorithm Performance Comparison",
                           ylabel="Fitness", save_path=None):
    """Create boxplot comparing multiple algorithms.
    
    Args:
        data_dict: dict of {algorithm_name: list of fitness values}
        title: plot title
        ylabel: y-axis label
        save_path: path to save figure
    """
    plt.figure(figsize=(12, 7))
    
    data = list(data_dict.values())
    labels = list(data_dict.keys())
    
    bp = plt.boxplot(data, labels=labels, patch_artist=True)
    
    # Color boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_bar_comparison(metrics_dict, metric_name="Mean Fitness", 
                       title="Algorithm Comparison", save_path=None):
    """Create bar chart comparing algorithms on a specific metric.
    
    Args:
        metrics_dict: dict of {algorithm_name: metric_value}
        metric_name: name of the metric
        title: plot title
        save_path: path to save figure
    """
    plt.figure(figsize=(12, 7))
    
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = plt.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title(title, fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel("Algorithm", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_tsp_route(cities, route, distance, title="TSP Solution", save_path=None, ax=None):
    """Plot TSP route on 2D city map.
    
    Args:
        cities: numpy array of city coordinates (n_cities, 2)
        route: list of city indices representing the tour
        distance: total tour distance
        title: plot title
        save_path: path to save figure
        ax: optional matplotlib axes to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
        show_plot = True
    else:
        show_plot = False

    # Plot cities
    ax.scatter(cities[:, 0], cities[:, 1], c='blue', s=100, zorder=3, label='Cities')
    
    # Plot route
    if route:
        route_cities = cities[route + [route[0]]]
        ax.plot(route_cities[:, 0], route_cities[:, 1], 
                'r-', linewidth=2, alpha=0.7, label='Route')
    
    # Add city labels
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y), fontsize=8, ha='center', va='center')
    
    ax.set_title(f"{title}\nTotal Distance: {distance:.2f}", fontsize=14)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def ensure_figure_dir(base_path="report/figures"):
    """Ensure the figures directory exists."""
    os.makedirs(base_path, exist_ok=True)
    return base_path

