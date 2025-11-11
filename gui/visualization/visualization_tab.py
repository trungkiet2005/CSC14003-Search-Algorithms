# File 2: visualization_tab.py - PyQt6 version
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QComboBox, QLineEdit, QScrollArea, QFrame, QGridLayout,
                              QTextEdit, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import threading
import os
import time
import numpy as np
import seaborn as sns

from .visualization_runner import VisualizationRunner
from utils.visualize import (
    plot_convergence_comparison, plot_boxplot_comparison,
    plot_parameter_sensitivity
)

ALGO_SPECIFIC_PARAMS = {
    'PSO': {
        'n_particles': {'label': 'Particle Count', 'default': 30, 'type': int, 'min': 5, 'max': 100},
        'w': {'label': 'Inertia Weight', 'default': 0.7298, 'type': float, 'min': 0.1, 'max': 1.5},
        'w_min': {'label': 'Min Inertia', 'default': 0.4, 'type': float, 'min': 0.1, 'max': 1.0},
        'w_max': {'label': 'Max Inertia', 'default': 0.9, 'type': float, 'min': 0.1, 'max': 1.5},
        'c1': {'label': 'Cognitive Coefficient', 'default': 1.49618, 'type': float, 'min': 0.0, 'max': 4.0},
        'c2': {'label': 'Social Coefficient', 'default': 1.49618, 'type': float, 'min': 0.0, 'max': 4.0},
        'v_max_ratio': {'label': 'Velocity Limit', 'default': 0.2, 'type': float, 'min': 0.05, 'max': 1.0},
    },
    'ABC': {
        'n_bees': {'label': 'Colony Size', 'default': 30, 'type': int, 'min': 5, 'max': 100},
        'limit': {'label': 'Scout Limit', 'default': None, 'type': int, 'min': 10, 'max': 1000, 'optional': True},
        'modification_rate': {'label': 'Modification Rate', 'default': 1.0, 'type': float, 'min': 0.1, 'max': 1.0},
    },
    'FA': {
        'n_fireflies': {'label': 'Population Size', 'default': 25, 'type': int, 'min': 5, 'max': 100},
        'alpha': {'label': 'Randomness', 'default': 0.5, 'type': float, 'min': 0.01, 'max': 2.0},
        'alpha_min': {'label': 'Min Randomness', 'default': 0.01, 'type': float, 'min': 0.001, 'max': 0.5},
        'beta0': {'label': 'Attractiveness', 'default': 1.0, 'type': float, 'min': 0.1, 'max': 5.0},
        'gamma': {'label': 'Absorption', 'default': 1.0, 'type': float, 'min': 0.01, 'max': 10.0},
    },
    'CS': {
        'n_nests': {'label': 'Host Nests', 'default': 25, 'type': int, 'min': 5, 'max': 100},
        'pa': {'label': 'Discovery Rate', 'default': 0.25, 'type': float, 'min': 0.0, 'max': 1.0},
        'beta': {'label': 'Lévy Parameter', 'default': 1.5, 'type': float, 'min': 0.5, 'max': 2.5},
        'step_size_factor': {'label': 'Step Scale', 'default': 0.01, 'type': float, 'min': 0.001, 'max': 0.5},
    }
}

class WorkerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    
    def __init__(self, algorithm, problem, dim, max_iter, n_runs, algo_params):
        super().__init__()
        self.algorithm = algorithm
        self.problem = problem
        self.dim = dim
        self.max_iter = max_iter
        self.n_runs = n_runs
        self.algo_params = algo_params
        self.cancel_flag = False
        
    def run(self):
        try:
            def progress_callback(msg):
                if self.cancel_flag:
                    raise KeyboardInterrupt("User cancelled")
                self.progress.emit(msg)
            
            runner = VisualizationRunner(seed=42)
            results = runner.run_visualization_analysis(
                self.algorithm, self.problem, self.dim, self.max_iter, 
                self.n_runs, self.algo_params, {}, progress_callback
            )
            
            if self.cancel_flag:
                raise KeyboardInterrupt("User cancelled")
            
            self.finished.emit(results)
            
        except KeyboardInterrupt:
            self.cancelled.emit()
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(str(e))
    
    def cancel(self):
        self.cancel_flag = True


class VisualizationTab:
    def __init__(self, parent):
        self.parent = parent
        self.is_running = False
        self.current_view = "convergence"
        self.metric_data = {}
        self.generated_figures = {}
        self.specific_param_entries = {}
        self.spinner_running = False
        self.worker = None
        
        # Setup layout
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        self._create_sidebar(layout)
        self._create_results_area(layout)
        self._update_specific_params_widgets()
        self._init_placeholder_text()
        
    def _create_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setFixedWidth(260)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
            }
        """)
        parent_layout.addWidget(sidebar)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Algorithm selection
        algo_label = QLabel("Algorithm")
        algo_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        algo_label.setStyleSheet("color: white;")
        layout.addWidget(algo_label)
        
        self.algo_menu = QComboBox()
        self.algo_menu.addItems(["PSO", "ABC", "FA", "CS"])
        self.algo_menu.currentTextChanged.connect(self._update_specific_params_widgets)
        self._style_combobox(self.algo_menu)
        layout.addWidget(self.algo_menu)
        
        # Problem selection
        problem_label = QLabel("Benchmark Function")
        problem_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        problem_label.setStyleSheet("color: white;")
        layout.addWidget(problem_label)
        
        self.problem_menu = QComboBox()
        self.problem_menu.addItems(["Sphere", "Rastrigin", "Rosenbrock", "Ackley"])
        self._style_combobox(self.problem_menu)
        layout.addWidget(self.problem_menu)
        
        # Parameters
        params_label = QLabel("Configuration")
        params_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        params_label.setStyleSheet("color: white; margin-top: 8px;")
        layout.addWidget(params_label)
        
        # Scrollable parameters area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background-color: #00A86B;
                border-radius: 5px;
            }
        """)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: #1e1e1e;")
        self.params_layout = QVBoxLayout(scroll_widget)
        self.params_layout.setSpacing(6)
        
        # Basic parameters
        self.dim_entry = self._create_input_field(self.params_layout, "Dimensions", "10")
        self.iter_entry = self._create_input_field(self.params_layout, "Max Iterations", "100")
        self.runs_entry = self._create_input_field(self.params_layout, "Number of Runs", "10")
        
        # Specific parameters containers
        self.specific_params_label = QLabel("Algorithm Parameters")
        self.specific_params_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.specific_params_label.setStyleSheet("color: white; margin-top: 8px;")
        
        self.specific_params_widget = QWidget()
        self.specific_params_layout = QVBoxLayout(self.specific_params_widget)
        self.specific_params_layout.setSpacing(5)
        
        self.params_layout.addWidget(self.specific_params_label)
        self.params_layout.addWidget(self.specific_params_widget)
        self.params_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # Action buttons
        self.run_button = QPushButton("▶ Launch")
        self.run_button.clicked.connect(self.run_experiment)
        self._style_button(self.run_button, "#00A86B", "#00D9A5")
        layout.addWidget(self.run_button)
        
        self.cancel_button = QPushButton("⏹ Cancel")
        self.cancel_button.clicked.connect(self.cancel_experiment)
        self._style_button(self.cancel_button, "#FF6B6B", "#FF4D4D")
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        
        self.save_button = QPushButton("💾 Export All")
        self.save_button.clicked.connect(self.save_all_figures)
        self._style_button(self.save_button, "#555555", "#666666")
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
        # Progress indicator
        self.progress_frame = QFrame()
        self.progress_frame.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border-radius: 6px;
                border: 1px solid #4a4a4a;
            }
        """)
        progress_layout = QHBoxLayout(self.progress_frame)
        
        self.spinner_label = QLabel("⏳")
        self.spinner_label.setFont(QFont("Segoe UI", 14))
        progress_layout.addWidget(self.spinner_label)
        
        self.progress_message = QLabel("")
        self.progress_message.setFont(QFont("Segoe UI", 9))
        self.progress_message.setStyleSheet("color: #aaaaaa;")
        self.progress_message.setWordWrap(True)
        progress_layout.addWidget(self.progress_message, 1)
        
        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)
        
        # Status
        self.status_label = QLabel("● Ready")
        self.status_label.setFont(QFont("Segoe UI", 10))
        self.status_label.setStyleSheet("color: #aaaaaa; padding: 8px;")
        layout.addWidget(self.status_label)
        
    def _create_header(self):
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border-radius: 6px;
            }
        """)
        layout = QVBoxLayout(header)
        
        title = QLabel("Algorithm Visualization")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Individual behavior analysis")
        subtitle.setFont(QFont("Segoe UI", 9))
        subtitle.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(subtitle)
        
        return header
        
    def _create_input_field(self, parent_layout, label_text, placeholder):
        label = QLabel(label_text)
        label.setFont(QFont("Segoe UI", 9))
        label.setStyleSheet("color: white;")
        parent_layout.addWidget(label)
        
        entry = QLineEdit()
        entry.setPlaceholderText(placeholder)
        entry.setText(placeholder)
        self._style_lineedit(entry)
        parent_layout.addWidget(entry)
        
        return entry
        
    def _style_combobox(self, combo):
        combo.setStyleSheet("""
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px;
                font-size: 10px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: white;
                selection-background-color: #00A86B;
            }
        """)
        
    def _style_lineedit(self, edit):
        edit.setStyleSheet("""
            QLineEdit {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 5px;
                font-size: 10px;
            }
            QLineEdit:focus {
                border: 1px solid #00A86B;
            }
        """)
        
    def _style_button(self, button, bg_color, hover_color):
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color};
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 11px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
            QPushButton:disabled {{
                background-color: #555555;
                color: #888888;
            }}
        """)
        
    def _update_specific_params_widgets(self, selected_algo=None):
        if selected_algo is None:
            selected_algo = self.algo_menu.currentText()
            
        # Clear existing widgets
        while self.specific_params_layout.count():
            child = self.specific_params_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.specific_param_entries = {}
        params = ALGO_SPECIFIC_PARAMS.get(selected_algo, {})
        
        if not params:
            self.specific_params_label.hide()
            self.specific_params_widget.hide()
            return
            
        self.specific_params_label.show()
        self.specific_params_widget.show()
        
        for name, config in params.items():
            label_text = config['label'] + (" (opt)" if config.get('optional') else "")
            label = QLabel(label_text)
            label.setFont(QFont("Segoe UI", 9))
            label.setStyleSheet("color: white;")
            self.specific_params_layout.addWidget(label)
            
            entry = QLineEdit()
            placeholder = str(config['default']) if config['default'] is not None else "auto"
            entry.setPlaceholderText(placeholder)
            self._style_lineedit(entry)
            self.specific_params_layout.addWidget(entry)
            
            self.specific_param_entries[name] = entry
            

    def _create_results_area(self, parent_layout):
        results = QFrame()
        results.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 8px;
                border: 1px solid #3a3a3a;
            }
        """)
        parent_layout.addWidget(results, 1)
        
        layout = QVBoxLayout(results)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Header
        header = QLabel("Interactive Results Dashboard")
        header.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        header.setStyleSheet("color: white;")
        layout.addWidget(header)
        
        # Metric buttons
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet("background-color: #3a3a3a; border-radius: 6px;")
        buttons_layout = QHBoxLayout(buttons_frame)
        
        self.convergence_btn = QPushButton("Convergence")
        self.performance_btn = QPushButton("Performance")
        self.sensitivity_btn = QPushButton("Sensitivity")
        self.landscape_btn = QPushButton("Landscape")
        
        for btn in [self.convergence_btn, self.performance_btn, self.sensitivity_btn, self.landscape_btn]:
            self._style_metric_button(btn)
            buttons_layout.addWidget(btn)
            
        self.convergence_btn.clicked.connect(lambda: self.show_metric_view("convergence"))
        self.performance_btn.clicked.connect(lambda: self.show_metric_view("performance"))
        self.sensitivity_btn.clicked.connect(lambda: self.show_metric_view("sensitivity"))
        self.landscape_btn.clicked.connect(lambda: self.show_metric_view("landscape"))
        
        layout.addWidget(buttons_frame)
        
        # Content area
        self.metric_content_frame = QFrame()
        self.metric_content_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border-radius: 6px;
                border: 1px solid #3a3a3a;
            }
        """)
        layout.addWidget(self.metric_content_frame, 1)
        
        self.metric_display_layout = QVBoxLayout(self.metric_content_frame)
        
    def _style_metric_button(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #888888;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                color: white;
            }
        """)
        
    def _init_placeholder_text(self):
        placeholder = "Select an algorithm and click 'Launch' to start."
        for view in ["convergence", "performance", "sensitivity", "landscape"]:
            self.metric_data[view] = placeholder
        self.show_metric_view(self.current_view)
        
    def show_metric_view(self, view_name):
        self.current_view = view_name
        
        # Clear existing widgets
        while self.metric_display_layout.count():
            child = self.metric_display_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Update button styles
        for btn in [self.convergence_btn, self.performance_btn, self.sensitivity_btn, self.landscape_btn]:
            self._style_metric_button(btn)
            
        # Highlight active button
        button_map = {
            "convergence": self.convergence_btn,
            "performance": self.performance_btn,
            "sensitivity": self.sensitivity_btn,
            "landscape": self.landscape_btn
        }
        if view_name in button_map:
            button_map[view_name].setStyleSheet("""
                QPushButton {
                    background-color: #00A86B;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    padding: 8px;
                    font-size: 10px;
                    font-weight: bold;
                }
            """)
        
        # Display content
        # Always regenerate the figure to avoid rendering issues with reused figure objects.
        fig = None
        data = self.metric_data.get(view_name)
        metadata = self.metric_data.get('metadata', {})

        # Check if we have actual results data to plot
        if isinstance(data, dict):
            plot_map = {
                "convergence": self._plot_convergence,
                "performance": self._plot_performance,
                "sensitivity": self._plot_sensitivity,
                "landscape": self._plot_landscape,
            }
            if view_name in plot_map:
                # Generate a new figure
                fig = plot_map[view_name](data, metadata)

        # If a figure was successfully generated, display it
        if fig:
            # Close the old figure for this view if it exists, to prevent memory leaks
            if view_name in self.generated_figures:
                plt.close(self.generated_figures[view_name])
            
            # Store the new figure and display it in a new canvas
            self.generated_figures[view_name] = fig
            canvas = FigureCanvas(fig)
            self.metric_display_layout.addWidget(canvas)
        else:
            # If no figure was generated (e.g., no data yet), show a placeholder text
            placeholder = data if data else "No data available."
            textbox = QTextEdit()
            textbox.setReadOnly(True)
            textbox.setText(str(placeholder))
            textbox.setStyleSheet("""
                QTextEdit {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 8px;
                    font-family: Consolas;
                    font-size: 9px;
                }
            """)
            self.metric_display_layout.addWidget(textbox)
            
    def _plot_convergence(self, data, metadata):
        fig, ax = plt.subplots(figsize=(10, 6))
        history = data['history']
        algo = metadata['algorithm']
        prob = metadata['problem']
        dim = metadata['dim']
        
        plot_convergence_comparison({algo: history}, title=f"Convergence: {algo} on {prob.capitalize()} (dim={dim})", log_scale=True, ax=ax)
        
        final_fitness, initial_fitness = history[-1], history[0]
        improvement = ((initial_fitness - final_fitness) / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
        textstr = f'Initial: {initial_fitness:.4f}\nFinal: {final_fitness:.4f}\nImprovement: {improvement:.2f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout(pad=0.5)
        return fig

    def _plot_performance(self, data, metadata):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        all_histories, best_fitnesses = data['all_histories'], data['best_fitnesses']
        algo, prob, n_runs = metadata['algorithm'], metadata['problem'], metadata['n_runs']
        
        fig.suptitle(f"Performance: {algo} on {prob.capitalize()} ({n_runs} runs)", fontsize=14, fontweight='bold')
        
        colors = sns.color_palette("husl", n_runs)
        for i, hist in enumerate(all_histories):
            ax1.plot(hist, alpha=0.5, linewidth=1, color=colors[i])
        ax1.plot(np.mean(all_histories, axis=0), 'k--', linewidth=2, alpha=0.8, label='Mean')
        ax1.set_title(f"Convergence Across {n_runs} Runs", fontsize=12)
        ax1.set_xlabel("Iteration", fontsize=10)
        ax1.set_ylabel("Fitness (log scale)", fontsize=10)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.2)
        ax1.legend(fontsize=8)
        
        plot_boxplot_comparison({algo: best_fitnesses}, title="Final Fitness Distribution", ax=ax2)
        mean_fit, std_fit = np.mean(best_fitnesses), np.std(best_fitnesses)
        stats_text = f'Mean: {mean_fit:.4f}\nStd: {std_fit:.4f}'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout(pad=0.5)
        return fig

    def _plot_sensitivity(self, data, metadata):
        fig, ax = plt.subplots(figsize=(10, 6))
        param_name = data['param_name']
        param_values = data['param_values']
        mean_fitness = data['mean_fitness']
        std_fitness = data['std_fitness']
        algo, prob = metadata['algorithm'], metadata['problem']
        
        plot_parameter_sensitivity(param_values, mean_fitness, std_fitness, 
                                   param_name.replace('_', ' ').title(), 
                                   title=f"Sensitivity: {algo} on {prob.capitalize()}",
                                   ax=ax)
        
        best_idx = np.argmin(mean_fitness)
        best_param, best_fitness = param_values[best_idx], mean_fitness[best_idx]
        ax.axvline(x=best_param, color='g', linestyle='--', lw=1.5, alpha=0.7, label='Best')
        ax.legend(fontsize=10)
        textstr = f'Optimal: {best_param}\nFitness: {best_fitness:.4f}'
        props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
        ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout(pad=0.5)
        return fig

    def _plot_landscape(self, data, metadata):
        fig = plt.figure(figsize=(12, 5))
        X, Y, Z = data['X'], data['Y'], data['Z']
        best_position = data['best_position']
        algo, prob = metadata['algorithm'], metadata['problem']
        
        fig.suptitle(f"Landscape: {algo} on {prob.capitalize()}", fontsize=14, fontweight='bold')
        
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, lw=0)
        if len(best_position) >= 2:
            z_best = np.interp(best_position[1], Y[:, 0], Z[:, np.argmin(np.abs(X[0, :] - best_position[0]))])
            ax1.scatter(best_position[0], best_position[1], z_best, c='r', s=150, marker='*', ec='k', lw=1, label=f'Solution', zorder=10)
            ax1.legend(fontsize=10)
        ax1.set_title('3D Surface', fontsize=12)
        ax1.tick_params(axis='both', which='major', labelsize=8)

        ax2 = fig.add_subplot(122)
        contour_plot = ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        fig.colorbar(contour_plot, ax=ax2)
        ax2.contour(X, Y, Z, levels=10, colors='k', alpha=0.3, linewidths=0.5)
        if len(best_position) >= 2:
            ax2.scatter(best_position[0], best_position[1], c='r', s=200, marker='*', ec='k', lw=1.5, label=f'Solution', zorder=10)
        ax2.set_title('Contour View', fontsize=12)
        ax2.set_aspect('equal')
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        plt.tight_layout(pad=0.5)
        return fig
        
    def _start_spinner(self):
        self.spinner_running = True
        self.spinner_chars = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self.spinner_index = 0
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self._animate_spinner)
        self.spinner_timer.start(100)
        
    def _stop_spinner(self):
        self.spinner_running = False
        if hasattr(self, 'spinner_timer'):
            self.spinner_timer.stop()
        self.spinner_label.setText("⏳")
        
    def _animate_spinner(self):
        if not self.spinner_running:
            return
        self.spinner_label.setText(self.spinner_chars[self.spinner_index])
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
        
    def _disable_inputs(self):
        self.algo_menu.setEnabled(False)
        self.problem_menu.setEnabled(False)
        self.dim_entry.setEnabled(False)
        self.iter_entry.setEnabled(False)
        self.runs_entry.setEnabled(False)
        for entry in self.specific_param_entries.values():
            entry.setEnabled(False)
            
    def _enable_inputs(self):
        self.algo_menu.setEnabled(True)
        self.problem_menu.setEnabled(True)
        self.dim_entry.setEnabled(True)
        self.iter_entry.setEnabled(True)
        self.runs_entry.setEnabled(True)
        for entry in self.specific_param_entries.values():
            entry.setEnabled(True)
            
    def update_status(self, message, color="#aaaaaa"):
        icon_map = {"Ready": "●", "Running": "⏳", "Complete": "✓", "Cancelled": "⏹", "Error": "✖"}
        icon = next((icon_map[key] for key in icon_map if key in message), "●")
        self.status_label.setText(f"{icon} {message}")
        self.status_label.setStyleSheet(f"color: {color}; padding: 8px;")
        
    def update_progress(self, message):
        self.progress_message.setText(message)
        
    def run_experiment(self):
        if self.is_running:
            return
            
        self.is_running = True
        self._disable_inputs()
        
        self.run_button.hide()
        self.cancel_button.show()
        self.save_button.setEnabled(False)
        self.progress_frame.show()
        
        self.update_progress("Initializing simulation...")
        self._start_spinner()
        self.update_status("Running...", "#FFA500")
        
        # Get parameters
        algorithm = self.algo_menu.currentText()
        problem = self.problem_menu.currentText().lower()
        dim = int(self.dim_entry.text() or 10)
        max_iter = int(self.iter_entry.text() or 100)
        n_runs = int(self.runs_entry.text() or 10)
        
        algo_params = {}
        for name, entry in self.specific_param_entries.items():
            val_str = entry.text()
            config = ALGO_SPECIFIC_PARAMS[algorithm][name]
            if val_str:
                try:
                    algo_params[name] = config['type'](val_str)
                except (ValueError, TypeError):
                    pass
            elif config['default'] is not None:
                algo_params[name] = config['default']
        
        problem_params = {}
        
        # Create and start worker thread
        self.worker = WorkerThread(algorithm, problem, dim, max_iter, n_runs, algo_params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self._update_ui_with_results)
        self.worker.error.connect(self._show_error)
        self.worker.cancelled.connect(self._experiment_cancelled)
        self.worker.finished.connect(self._experiment_complete)
        self.worker.error.connect(self._experiment_complete)
        self.worker.cancelled.connect(self._experiment_complete)
        self.worker.start()
        
    def cancel_experiment(self):
        if not self.is_running or not self.worker:
            return
        
        self.worker.cancel()
        self.update_progress("Cancelling...")
        self.update_status("Cancelling...", "#FF6B6B")
        self.cancel_button.setEnabled(False)
        
    def _clear_figures(self):
        for fig in self.generated_figures.values():
            plt.close(fig)
        self.generated_figures.clear()
        
    def _update_ui_with_results(self, results):
        self._clear_figures()
        self.metric_data = results
        self.show_metric_view(self.current_view)
        self.save_button.setEnabled(True)
        
    def _show_error(self, error_msg):
        self._clear_figures()
        error_text = f"✖ SIMULATION ERROR\n\n{error_msg}"
        self.metric_data = {k: error_text for k in ["convergence", "performance", "sensitivity", "landscape"]}
        self.show_metric_view(self.current_view)
        self.update_status("Error occurred", "#FF6B6B")
        
    def _experiment_cancelled(self):
        self._clear_figures()
        self.update_progress("Cancelled by user")
        self.update_status("Cancelled", "#FF6B6B")
        self.metric_data = {k: "⏹ Simulation cancelled by user." for k in ["convergence", "performance", "sensitivity", "landscape"]}
        self.show_metric_view(self.current_view)
        
    def _experiment_complete(self):
        self.is_running = False
        self._enable_inputs()
        self._stop_spinner()
        self.progress_frame.hide()
        self.cancel_button.hide()
        self.run_button.show()
        self.cancel_button.setEnabled(True)
        
        if "Cancelled" not in self.status_label.text() and "Error" not in self.status_label.text():
            self.update_status("Simulation Complete", "#00D9A5")
        
    def save_all_figures(self):
        if not isinstance(self.metric_data.get('metadata'), dict):
            self.update_status("No results to export", "#FFD166")
            return

        metadata = self.metric_data['metadata']
        algo = metadata['algorithm']
        prob = metadata['problem']
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir_str = f"results/figures/visualization/{algo}_{prob}_{timestamp}"
        output_dir = os.path.join(os.getcwd(), output_dir_str.replace('/', os.sep))
        os.makedirs(output_dir, exist_ok=True)

        views_to_save = ["convergence", "performance", "sensitivity", "landscape"]
        
        plot_map = {
            "convergence": self._plot_convergence,
            "performance": self._plot_performance,
            "sensitivity": self._plot_sensitivity,
            "landscape": self._plot_landscape,
        }

        saved_files = []
        for view in views_to_save:
            data = self.metric_data.get(view)
            if not isinstance(data, dict):
                continue

            # Always regenerate the figure to ensure it's valid
            fig_to_save = None
            if view in plot_map:
                fig_to_save = plot_map[view](data, metadata)

            if fig_to_save:
                try:
                    filepath = os.path.join(output_dir, f"{view}.png")
                    fig_to_save.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_files.append(os.path.basename(filepath))
                finally:
                    plt.close(fig_to_save)  # Ensure figure is closed to free memory


        if saved_files:
            self.update_status(f"Exported {len(saved_files)} plots to {output_dir_str}", "#00D9A5")
        else:
            self.update_status("No valid plots to export", "#FFD166")