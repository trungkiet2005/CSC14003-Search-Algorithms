from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QComboBox, QLineEdit, QScrollArea, QFrame, QGridLayout,
                              QTextEdit, QFileDialog, QCheckBox)
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
    plot_parameter_sensitivity, plot_3d_surface, plot_contour
)
from problems.continuous import get_problem
from config.experiment_config import (
    PARAMETER_RANGES, ALGORITHM_UI_CONFIG, ExperimentConfig, 
    ProblemConfig, AlgorithmConfig
)

class WorkerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    
    def __init__(self, exp_config: ExperimentConfig, sensitivity_params: list, seed: int):
        super().__init__()
        self.exp_config = exp_config
        self.sensitivity_params = sensitivity_params
        self.seed = seed
        self.cancel_flag = False
        
    def run(self):
        try:
            def progress_callback(msg):
                if self.cancel_flag:
                    raise KeyboardInterrupt("User cancelled")
                self.progress.emit(msg)
            
            runner = VisualizationRunner(seed=self.seed)
            # Pass the entire ExperimentConfig object to the runner
            results = runner.run_visualization_analysis(
                self.exp_config, self.sensitivity_params, progress_callback
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
        self.sensitivity_param_checkboxes = {}
        self.spinner_running = False
        self.worker = None
        self.problem_seeds = {}
        
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
        self.algo_menu.addItems(["PSO", "ABC", "FA", "CS", "ACO"])
        self.algo_menu.currentTextChanged.connect(self._handle_algorithm_selection)
        self._style_combobox(self.algo_menu)
        layout.addWidget(self.algo_menu)
        
        # Problem selection
        problem_label = QLabel("Benchmark Function")
        problem_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        problem_label.setStyleSheet("color: white;")
        layout.addWidget(problem_label)
        
        self.problem_menu = QComboBox()
        self.problem_menu.addItems(["Sphere", "Rastrigin", "Rosenbrock", "Ackley", "TSP"])
        self.problem_menu.currentTextChanged.connect(self._handle_problem_selection)
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
        self.dim_label = QLabel("Dimensions")
        self.dim_label.setFont(QFont("Segoe UI", 9))
        self.dim_label.setStyleSheet("color: white;")
        self.params_layout.addWidget(self.dim_label)
        
        self.dim_entry = QLineEdit()
        self.dim_entry.setPlaceholderText("10")
        self.dim_entry.setText("10")
        self._style_lineedit(self.dim_entry)
        self.params_layout.addWidget(self.dim_entry)

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

        # Sensitivity analysis parameters
        self.sensitivity_params_label = QLabel("Sensitivity Analysis Parameters")
        self.sensitivity_params_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.sensitivity_params_label.setStyleSheet("color: white; margin-top: 8px;")

        self.sensitivity_params_widget = QWidget()
        self.sensitivity_params_layout = QVBoxLayout(self.sensitivity_params_widget)
        self.sensitivity_params_layout.setSpacing(5)

        self.params_layout.addWidget(self.sensitivity_params_label)
        self.params_layout.addWidget(self.sensitivity_params_widget)
        
        self.params_layout.addStretch()
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Seed input
        seed_label = QLabel("Experiment Seed")
        seed_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        seed_label.setStyleSheet("color: white; margin-top: 8px;")
        layout.addWidget(seed_label)

        self.seed_entry = QLineEdit()
        self.seed_entry.setPlaceholderText("Leave empty for random")
        self._style_lineedit(self.seed_entry)
        layout.addWidget(self.seed_entry)
        
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
        
        # Add a reset seed button
        self.reset_seed_button = QPushButton("🔄 Reset Seed")
        self.reset_seed_button.clicked.connect(self._reset_seed)
        self._style_button(self.reset_seed_button, "#555555", "#666666")
        layout.addWidget(self.reset_seed_button)

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
            
        # Clear existing widgets from both layouts
        for layout in [self.specific_params_layout, self.sensitivity_params_layout]:
            while layout.count():
                child = layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        
        self.specific_param_entries = {}
        self.sensitivity_param_checkboxes = {}

        # --- Populate Algorithm-specific parameters ---
        params = ALGORITHM_UI_CONFIG.get(selected_algo, {})
        if not params:
            self.specific_params_label.hide()
            self.specific_params_widget.hide()
        else:
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

        # --- Populate Sensitivity analysis parameters ---
        sensitivity_params = PARAMETER_RANGES.get(selected_algo, {})
        if not sensitivity_params:
            self.sensitivity_params_label.hide()
            self.sensitivity_params_widget.hide()
        else:
            self.sensitivity_params_label.show()
            self.sensitivity_params_widget.show()
            algo_ui_config = ALGORITHM_UI_CONFIG.get(selected_algo, {})
            for param_name, _ in sensitivity_params.items():
                # Get the descriptive label from ALGORITHM_UI_CONFIG, default to param_name if not found
                label = algo_ui_config.get(param_name, {}).get('label', param_name)
                checkbox = QCheckBox(label)
                checkbox.setStyleSheet("color: white;")
                self.sensitivity_params_layout.addWidget(checkbox)
                # The key in the dictionary remains the internal parameter name
                self.sensitivity_param_checkboxes[param_name] = checkbox
            

    def _handle_algorithm_selection(self, algorithm):
        self._update_specific_params_widgets(algorithm)
        
        is_aco = (algorithm == "ACO")
        
        # Block signals to prevent infinite loops
        self.problem_menu.blockSignals(True)
        
        if is_aco:
            self.problem_menu.setCurrentText("TSP")
            self.problem_menu.setEnabled(False)
            self.dim_label.setText("Number of Cities")
            self.dim_entry.setText("50") # Default city count for TSP
            self.landscape_btn.setEnabled(False)
        else:
            if self.problem_menu.currentText() == "TSP":
                self.problem_menu.setCurrentText("Sphere")
            self.problem_menu.setEnabled(True)
            self.dim_label.setText("Dimensions")
            self.landscape_btn.setEnabled(True)
            
        self.problem_menu.blockSignals(False)

    def _handle_problem_selection(self, problem):
        is_tsp = (problem == "TSP")
        
        # Block signals to prevent infinite loops
        self.algo_menu.blockSignals(True)
        
        if is_tsp:
            self.algo_menu.setCurrentText("ACO")
            self.algo_menu.setEnabled(False)
            self.dim_label.setText("Number of Cities")
            self.dim_entry.setText("50") # Default city count for TSP
            self.landscape_btn.setEnabled(False)
        else:
            if self.algo_menu.currentText() == "ACO":
                self.algo_menu.setCurrentText("PSO")
            self.algo_menu.setEnabled(True)
            self.dim_label.setText("Dimensions")
            self.landscape_btn.setEnabled(True)
            
        self.algo_menu.blockSignals(False)
        # We still need to update params when algorithm changes
        self._update_specific_params_widgets()
            
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
            placeholder = data if data else "Choose the parameters that you want to examine in the sidebar."
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
        
        y_label = "Distance" if prob == 'tsp' else "Fitness"
        log_scale = False if prob == 'tsp' else True
        
        plot_convergence_comparison({algo: history}, title=f"Convergence: {algo} on {prob.capitalize()} (dim={dim})", log_scale=log_scale, ax=ax, ylabel=y_label)
        
        final_val, initial_val = history[-1], history[0]
        improvement = ((initial_val - final_val) / abs(initial_val)) * 100 if initial_val != 0 else 0
        
        val_name = "Dist" if prob == 'tsp' else "Fit"
        textstr = f'Initial {val_name}: {initial_val:.4f}\nFinal {val_name}: {final_val:.4f}\nImprovement: {improvement:.2f}%'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout(pad=0.5)
        return fig

    def _plot_performance(self, data, metadata):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        all_histories, best_fitnesses = data['all_histories'], data['best_fitnesses']
        algo, prob, n_runs = metadata['algorithm'], metadata['problem'], metadata['n_runs']
        
        y_label = "Distance" if prob == 'tsp' else "Fitness"
        log_scale = False if prob == 'tsp' else True
        
        fig.suptitle(f"Performance: {algo} on {prob.capitalize()} ({n_runs} runs)", fontsize=14, fontweight='bold')
        
        colors = sns.color_palette("husl", n_runs)
        for i, hist in enumerate(all_histories):
            ax1.plot(hist, alpha=0.5, linewidth=1, color=colors[i])
        ax1.plot(np.mean(all_histories, axis=0), 'k--', linewidth=2, alpha=0.8, label='Mean')
        ax1.set_title(f"Convergence Across {n_runs} Runs", fontsize=12)
        ax1.set_xlabel("Iteration", fontsize=10)
        ax1.set_ylabel(f"{y_label}{' (log scale)' if log_scale else ''}", fontsize=10)
        if log_scale:
            ax1.set_yscale('log')
        ax1.grid(True, alpha=0.2)
        ax1.legend(fontsize=8)
        
        # Add mean fitness to the line plot
        mean_fit = np.mean(best_fitnesses)
        stats_text_ax1 = f'Mean Final: {mean_fit:.4f}'
        props_ax1 = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        ax1.text(0.98, 0.95, stats_text_ax1, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right', bbox=props_ax1)
        
        boxplot_title = f"Final {y_label} Distribution"
        plot_boxplot_comparison({algo: best_fitnesses}, title=boxplot_title, ax=ax2, ylabel=f"Final {y_label}")
        std_fit = np.std(best_fitnesses)
        stats_text_ax2 = f'Mean: {mean_fit:.4f}\nStd: {std_fit:.4f}'
        props_ax2 = dict(boxstyle='round', facecolor='lightblue', alpha=0.7)
        ax2.text(0.98  , 0.95, stats_text_ax2, transform=ax2.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right', bbox=props_ax2)
        
        plt.tight_layout(pad=0.5)
        return fig

    def _plot_sensitivity(self, data, metadata):
        if not data:
            return None

        num_params = len(data)
        if num_params == 0:
            return None

        # Determine grid size
        cols = int(np.ceil(np.sqrt(num_params)))
        rows = int(np.ceil(num_params / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
        axes = axes.flatten()

        algo, prob = metadata['algorithm'], metadata['problem']
        fig.suptitle(f"Sensitivity Analysis: {algo} on {prob.capitalize()}", fontsize=16, fontweight='bold')

        param_names = list(data.keys())
        
        # Get the UI config for the current algorithm
        algo_ui_config = ALGORITHM_UI_CONFIG.get(algo, {})

        y_label = "Distance" if prob == 'tsp' else "Fitness"

        for i in range(num_params):
            param_name = param_names[i]
            param_data = data[param_name]
            ax = axes[i]

            param_values = param_data['param_values']
            mean_fitness = param_data['mean_fitness']
            std_fitness = param_data['std_fitness']

            # Get the descriptive label, default to a formatted version of the param_name
            param_label = algo_ui_config.get(param_name, {}).get('label', param_name.replace('_', ' ').title())

            plot_parameter_sensitivity(param_values, mean_fitness, std_fitness, 
                                       param_label, 
                                       title=f"Parameter: {param_label}",
                                       ax=ax,
                                       ylabel=y_label)
            
            best_idx = np.argmin(mean_fitness)
            best_param, best_fitness = param_values[best_idx], mean_fitness[best_idx]
            ax.axvline(x=best_param, color='g', linestyle='--', lw=1.5, alpha=0.7, label=f'Best: {best_param:.2f}')
            ax.legend(fontsize=8)
            
        # Hide unused subplots
        for i in range(num_params, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig

    def _plot_landscape(self, data, metadata):
        if metadata['problem'] == 'tsp':
            return None
        problem_func, problem_info = get_problem(metadata['problem'], dim=2)
        bounds = problem_info['bounds']
        best_position = data.get('best_position')
        best_fitness = data.get('best_fitness')
        
        fig = plt.figure(figsize=(12, 5.5))
        fig.suptitle(f"Landscape: {metadata['algorithm']} on {metadata['problem'].capitalize()}", fontsize=14, fontweight='bold')

        # 3D Surface Plot
        ax1 = fig.add_subplot(121, projection='3d')
        plot_3d_surface(
            func=problem_func,
            bounds=bounds,
            title='3D Surface View',
            best_point=best_position,
            best_fitness=best_fitness,
            ax=ax1
        )

        # Contour Plot
        ax2 = fig.add_subplot(122)
        plot_contour(
            func=problem_func,
            bounds=bounds,
            title='Contour View',
            best_point=best_position,
            ax=ax2
        )
        ax2.set_aspect('equal')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        self.seed_entry.setEnabled(False)
        self.reset_seed_button.setEnabled(False)
        for entry in self.specific_param_entries.values():
            entry.setEnabled(False)
            
    def _enable_inputs(self):
        self.algo_menu.setEnabled(True)
        self.problem_menu.setEnabled(True)
        self.dim_entry.setEnabled(True)
        self.iter_entry.setEnabled(True)
        self.runs_entry.setEnabled(True)
        self.seed_entry.setEnabled(True)
        self.reset_seed_button.setEnabled(True)
        for entry in self.specific_param_entries.values():
            entry.setEnabled(True)
            
    def update_status(self, message, color="#aaaaaa"):
        icon_map = {"Ready": "●", "Running": "⏳", "Complete": "✓", "Cancelled": "⏹", "Error": "✖"}
        icon = next((icon_map[key] for key in icon_map if key in message), "●")
        self.status_label.setText(f"{icon} {message}")
        self.status_label.setStyleSheet(f"color: {color}; padding: 8px;")
        
    def update_progress(self, message):
        self.progress_message.setText(message)
        
    def _get_experiment_config(self, seed) -> ExperimentConfig:
        """Gathers all settings from the UI and builds an ExperimentConfig object."""
        algorithm_name = self.algo_menu.currentText()
        problem_name = self.problem_menu.currentText().lower()
        
        # --- Algorithm Configuration ---
        algo_params = {}
        for name, entry in self.specific_param_entries.items():
            val_str = entry.text()
            config = ALGORITHM_UI_CONFIG[algorithm_name][name]
            if val_str:
                try:
                    algo_params[name] = config['type'](val_str)
                except (ValueError, TypeError):
                    # Use default if conversion fails
                    if config['default'] is not None:
                        algo_params[name] = config['default']
            elif config['default'] is not None:
                algo_params[name] = config['default']
        
        algorithm_config = AlgorithmConfig(name=algorithm_name, params=algo_params)

        # --- Problem Configuration ---
        if problem_name == 'tsp':
            problem_config = ProblemConfig(
                name='tsp',
                dim=int(self.dim_entry.text() or 50), # Number of cities for TSP
                max_iter=int(self.iter_entry.text() or 100)
            )
        else:
            problem_config = ProblemConfig(
                name=problem_name,
                dim=int(self.dim_entry.text() or 10),
                max_iter=int(self.iter_entry.text() or 100)
            )

        # --- Experiment Configuration ---
        exp_config = ExperimentConfig(
            name=f"vis_{algorithm_name}_on_{problem_name}",
            problem=problem_config,
            algorithms=[algorithm_config],
            n_runs=int(self.runs_entry.text() or 10),
            seed=seed,
            output_dir="results/gui_runs"
        )
        
        return exp_config

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
        
        # --- Determine Seed ---
        problem_key = f"{self.problem_menu.currentText().lower()}_{int(self.dim_entry.text() or 10)}"
        seed_text = self.seed_entry.text()
        seed = None

        if seed_text.strip().isdigit():
            seed = int(seed_text)
            self.update_progress(f"Using manually entered seed: {seed}")
        else:
            if problem_key not in self.problem_seeds:
                self.problem_seeds[problem_key] = np.random.randint(0, 2**31 - 1)
            seed = self.problem_seeds[problem_key]
            self.update_progress(f"Using session seed {seed} for {problem_key.split('_')[0].capitalize()} (dim={problem_key.split('_')[1]})")

        # Update the UI to show the seed being used
        self.seed_entry.setText(str(seed))

        # Create the experiment config from the UI
        exp_config = self._get_experiment_config(seed)
        
        # Get sensitivity analysis parameters
        sensitivity_params = [name for name, checkbox in self.sensitivity_param_checkboxes.items() if checkbox.isChecked()]
        
        # Create and start worker thread
        self.worker = WorkerThread(exp_config, sensitivity_params, seed)
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
        
    def _reset_seed(self):
        self.seed_entry.clear()
        problem = self.problem_menu.currentText().lower()
        dim = int(self.dim_entry.text() or 10)
        problem_key = f"{problem}_{dim}"
        if problem_key in self.problem_seeds:
            del self.problem_seeds[problem_key]
            self.update_status(f"Seed reset for {problem.capitalize()} (dim={dim}). New seed will be generated.", "#00D9A5")
        else:
            self.update_status("Seed cleared. New seed will be generated.", "#FFD166")

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