from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.benchmark import AlgorithmStats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import pandas as pd


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300


def ensure_figure_dir(base_path: str = "report/figures") -> Path:
    """Ensure the figures directory exists"""
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_convergence_comparison(histories_dict: Dict[str, List[float]], 
                               title: str = "Convergence Comparison",
                               save_path: Optional[str] = None,
                               log_scale: bool = False,
                               xlabel: str = "Iteration",
                               ylabel: str = "Fitness",
                               ax: Optional[plt.Axes] = None,
                               final_fitness_values: Optional[Dict[str, float]] = None) -> None:
    """
    Compare convergence of multiple algorithms
    
    Args:
        histories_dict: Dictionary of {algorithm_name: history}
        title: Plot title
        save_path: Path to save figure
        log_scale: Use log scale for y-axis
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        ax: Matplotlib axes (if None, creates new figure)
        final_fitness_values: Optional dict of {algorithm_name: final_fitness} to annotate plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        show_plot = True
    else:
        show_plot = False
    
    colors = sns.color_palette("husl", len(histories_dict))
    
    for (name, history), color in zip(histories_dict.items(), colors):
        ax.plot(history, label=name, linewidth=2.5, alpha=0.8, color=color)
        if final_fitness_values and name in final_fitness_values:
            final_fitness = final_fitness_values[name]
            ax.text(len(history) - 1, history[-1], f' {final_fitness:.4e}', color=color,
                    verticalalignment='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=color, lw=0.5, alpha=0.7))

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel + (" (log scale)" if log_scale else ""), fontsize=13)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
        # plt.show()


def plot_boxplot_comparison(data_dict: Dict[str, List[float]],
                           title: str = "Algorithm Performance Distribution",
                           ylabel: str = "Final Fitness",
                           save_path: Optional[str] = None,
                           ax: Optional[plt.Axes] = None) -> None:
    """
    Create boxplot comparing multiple algorithms
    
    Args:
        data_dict: Dictionary of {algorithm_name: list of fitness values}
        title: Plot title
        ylabel: Y-axis label
        save_path: Path to save figure
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        show_plot = True
    else:
        show_plot = False
    
    # Prepare data
    data = list(data_dict.values())
    labels = list(data_dict.keys())
    
    # Create boxplot
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(color='blue', linewidth=2, linestyle='--'))
    
    # Add annotations for medians
    for i, (name, values) in enumerate(data_dict.items()):
        median = np.median(values)
        ax.text(i + 1, median, f'{median:.4e}',
                horizontalalignment='center',
                verticalalignment='bottom',
                fontdict={'size': 9, 'color': 'black', 'weight': 'semibold'},
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', lw=0.5, alpha=0.6))

    # Color boxes
    colors = sns.color_palette("Set3", len(data))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Algorithm", fontsize=13)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add legend for mean and median
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=2, label='Median'),
        Line2D([0], [0], color='blue', linewidth=2, linestyle='--', label='Mean')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
        # plt.show()


def plot_3d_surface(func: Callable, bounds: Tuple[float, float],
                   resolution: int = 50,
                   title: str = "3D Function Surface",
                   save_path: Optional[str] = None,
                   best_point: Optional[np.ndarray] = None,
                   best_fitness: Optional[float] = None,
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """
    Plot 3D surface of a 2D objective function
    
    Args:
        func: Objective function (takes 2D array)
        bounds: Tuple (lower, upper) for both dimensions
        resolution: Number of points per dimension
        title: Plot title
        save_path: Path to save figure
        best_point: Optional (x, y) point to mark on surface
        best_fitness: Optional fitness value at the best point
        ax: Optional Matplotlib axes to plot on.
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
    
    # Create 3D plot if no axes are provided
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        show_plot = True
    else:
        fig = ax.get_figure()
        show_plot = False

    # Surface plot with improved colors
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9,
                          linewidth=0, antialiased=True,
                          edgecolor='none')
    
    # Mark best point if provided
    if best_point is not None and len(best_point) >= 2:
        z_best = best_fitness if best_fitness is not None else func(best_point)
        ax.scatter([best_point[0]], [best_point[1]], [z_best],
                  color='red', s=200, marker='*', 
                  edgecolors='black', linewidths=2,
                  label='Best Solution', zorder=10)
        
        # Add text annotation for the best point
        pos_text = f"Pos: [{best_point[0]:.2e}, {best_point[1]:.2e}]"
        fit_text = f"Fit: {z_best:.2e}"
        annotation_text = f"{pos_text}\n{fit_text}"
        
        ax.text(best_point[0], best_point[1], z_best, annotation_text,
                color='black', zorder=11, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', fc='yellow', ec='black', lw=1, alpha=0.8))
        
        ax.legend(fontsize=12)
    
    ax.set_xlabel('X', fontsize=12, labelpad=10)
    ax.set_ylabel('Y', fontsize=12, labelpad=10)
    ax.set_zlabel('f(X, Y)', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar if we created the figure
    if show_plot:
        fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    
    # Adjust viewing angle
    ax.view_init(elev=30, azim=45)
    
    if show_plot:
        plt.tight_layout()
    
    if save_path and show_plot:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    return fig


def plot_contour(func: Callable, bounds: Tuple[float, float],
                resolution: int = 100,
                title: str = "Contour Plot",
                save_path: Optional[str] = None,
                best_point: Optional[np.ndarray] = None,
                particle_positions: Optional[np.ndarray] = None,
                ax: Optional[plt.Axes] = None) -> None:
    """
    Plot contour of a 2D objective function
    
    Args:
        func: Objective function
        bounds: Tuple (lower, upper)
        resolution: Number of points per dimension
        title: Plot title
        save_path: Path to save figure
        best_point: Optional (x, y) point to mark
        particle_positions: Optional array of particle positions to plot
        ax: Matplotlib axes
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        show_plot = True
    else:
        fig = ax.get_figure()
        show_plot = False
    
    # Contour plot with filled contours
    contourf = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
    contour = ax.contour(X, Y, Z, levels=15, colors='black', 
                         alpha=0.3, linewidths=0.5)
    
    # Colorbar
    cbar = fig.colorbar(contourf, ax=ax, label='Fitness')
    
    # Plot particles if provided
    if particle_positions is not None:
        positions = np.array(particle_positions)
        if positions.shape[1] >= 2:
            ax.scatter(positions[:, 0], positions[:, 1],
                      c='cyan', s=40, alpha=0.7, 
                      edgecolors='black', linewidths=0.5,
                      label='Particles', zorder=5)
    
    # Mark best point
    if best_point is not None and len(best_point) >= 2:
        ax.scatter([best_point[0]], [best_point[1]],
                  color='red', s=300, marker='*',
                  edgecolors='black', linewidths=2,
                  label='Best Solution', zorder=10)
    
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
        # plt.show()


def plot_tsp_route(cities: np.ndarray, route: list, distance: float,
                  title: str = "TSP Solution",
                  save_path: Optional[str] = None,
                  ax: Optional[plt.Axes] = None) -> None:
    """
    Plot TSP route on 2D city map
    
    Args:
        cities: Array of city coordinates (n_cities, 2)
        route: List of city indices representing the tour
        distance: Total tour distance
        title: Plot title
        save_path: Path to save figure
        ax: Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
        show_plot = True
    else:
        show_plot = False
    
    # Plot route
    if route:
        route_cities = cities[route + [route[0]]]
        ax.plot(route_cities[:, 0], route_cities[:, 1],
               'b-', linewidth=2, alpha=0.6, label='Route', zorder=1)
        
        # Add arrows to show direction
        for i in range(len(route)):
            start = cities[route[i]]
            end = cities[route[(i + 1) % len(route)]]
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', color='blue',
                                     lw=1.5, alpha=0.5),
                       zorder=1)
    
    # Plot cities
    ax.scatter(cities[:, 0], cities[:, 1],
              c='red', s=150, zorder=3,
              edgecolors='black', linewidths=1.5,
              label='Cities')
    
    # Add city labels
    for i, (x, y) in enumerate(cities):
        ax.annotate(str(i), (x, y),
                   fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold',
                   zorder=4)
    
    # Highlight start/end city
    if route:
        start_city = cities[route[0]]
        ax.scatter([start_city[0]], [start_city[1]],
                  c='green', s=250, marker='s',
                  edgecolors='black', linewidths=2,
                  label='Start/End', zorder=5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
        # plt.show()


def plot_parameter_sensitivity(param_values: List, 
                               mean_fitness: List[float],
                               std_fitness: List[float],
                               param_name: str,
                               title: str = "Parameter Sensitivity Analysis",
                               save_path: Optional[str] = None,
                               ax: Optional[plt.Axes] = None,
                               ylabel: str = "Fitness") -> None:
    """
    Plot parameter sensitivity analysis
    
    Args:
        param_values: List of parameter values tested
        mean_fitness: Mean fitness for each parameter value
        std_fitness: Standard deviation for each parameter value
        param_name: Name of the parameter
        title: Plot title
        save_path: Path to save figure
        ax: Matplotlib axes
        ylabel: Y-axis label
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
        show_plot = True
    else:
        show_plot = False

    # Plot mean with error bars
    ax.errorbar(param_values, mean_fitness, yerr=std_fitness,
               marker='o', markersize=8, linewidth=2,
               capsize=5, capthick=2, alpha=0.8,
               label='Mean ± Std')
    
    ax.set_xlabel(param_name, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    if show_plot:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved: {save_path}")
    
    # plt.show()


def plot_comparison_table(df: pd.DataFrame, 
                          title: str = "Algorithm Comparison",
                          save_path: Optional[str] = None) -> None:
    """
    Create a visual comparison table
    
    Args:
        df: DataFrame with comparison results
        title: Table title
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Select key columns
    columns = ['algorithm_name', 'mean_fitness', 'std_fitness', 
               'mean_time', 'convergence_rate']
    display_df = df[columns].copy()
    
    # Rename columns for display
    display_df.columns = ['Algorithm', 'Mean Fitness', 'Std Fitness',
                         'Mean Time (s)', 'Conv. Rate']
    
    # Format numbers
    display_df['Mean Fitness'] = display_df['Mean Fitness'].apply(lambda x: f'{x:.6f}')
    display_df['Std Fitness'] = display_df['Std Fitness'].apply(lambda x: f'{x:.6f}')
    display_df['Mean Time (s)'] = display_df['Mean Time (s)'].apply(lambda x: f'{x:.3f}')
    display_df['Conv. Rate'] = display_df['Conv. Rate'].apply(
        lambda x: f'{x:.2%}' if pd.notna(x) else 'N/A'
    )
    
    # Create table
    table = ax.table(cellText=display_df.values,
                    colLabels=display_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")
    
    # plt.show()


def plot_scalability_comparison(scalability_data: Dict[str, Dict[str, List]],
                                    title: str = "Scalability Analysis") -> plt.Figure:
    """
    Plot scalability of algorithms.
    
    Args:
        scalability_data: Dict of {algo_name: {'dims': [...], 'fitness': [...], 'times': [...]}}
        title: Main title for the figure.
        
    Returns:
        A matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=18, fontweight='bold')
    
    colors = sns.color_palette("husl", len(scalability_data))
    
    # 1. Fitness vs. Dimensions
    for (algo_name, data), color in zip(scalability_data.items(), colors):
        ax1.plot(data['dims'], data['fitness'], marker='o', linestyle='-', color=color, label=algo_name)
    ax1.set_title("Fitness vs. Problem Size", fontsize=14)
    ax1.set_xlabel("Problem Dimension", fontsize=12)
    ax1.set_ylabel("Best Fitness", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 2. Time vs. Dimensions
    for (algo_name, data), color in zip(scalability_data.items(), colors):
        ax2.plot(data['dims'], data['times'], marker='o', linestyle='-', color=color, label=algo_name)
    ax2.set_title("Execution Time vs. Problem Size", fontsize=14)
    ax2.set_xlabel("Problem Dimension", fontsize=12)
    ax2.set_ylabel("Mean Time (seconds)", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig



def plot_complexity_comparison(stats_list: List,
                               title: str = "Computational Complexity Comparison",
                               ) -> plt.Figure:
    """
    Create bar charts comparing execution time and memory usage.
    
    Args:
        stats_list: List of AlgorithmStats objects from benchmark runner.
        title: The main title for the figure.
        
    Returns:
        A matplotlib Figure object containing the plots.
    """
    algo_names = [s.algorithm_name for s in stats_list]
    mean_times = [s.mean_time for s in stats_list]
    mean_mems = [getattr(s, 'mean_mem', 0.0) for s in stats_list]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("viridis", len(algo_names))
    
    # 1. Execution Time Plot
    bars1 = ax1.bar(algo_names, mean_times, color=colors, alpha=0.8)
    ax1.set_title("Mean Execution Time", fontsize=14)
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    # Add value annotations for time
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', 
                 va='bottom', ha='center', fontsize=9)

    # 2. Memory Usage Plot
    bars2 = ax2.bar(algo_names, mean_mems, color=colors, alpha=0.8)
    ax2.set_title("Mean Peak Memory Usage", fontsize=14)
    ax2.set_ylabel("Memory (MB)", fontsize=12)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add value annotations for memory
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}MB', 
                 va='bottom', ha='center', fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig