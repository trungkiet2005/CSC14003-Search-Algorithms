from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                              QComboBox, QLineEdit, QScrollArea, QFrame, QTextEdit,
                              QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
import os
import time
import numpy as np
import seaborn as sns

from .comparison_runner import ComparisonRunner
from utils.visualize import (
    plot_convergence_comparison, plot_boxplot_comparison,
    plot_complexity_comparison, plot_scalability_comparison,
    plot_tsp_route
)
from config.experiment_config import (
    ExperimentConfig, ProblemConfig, AlgorithmConfig
)


class ComparisonWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()
    
    def __init__(self, exp_config: ExperimentConfig, seed: int):
        super().__init__()
        self.exp_config = exp_config
        self.seed = seed
        self.cancel_flag = False
        
    def run(self):
        try:
            def progress_callback(msg):
                if self.cancel_flag:
                    raise KeyboardInterrupt("User cancelled")
                self.progress.emit(msg)
            
            runner = ComparisonRunner(seed=self.seed)
            
            if self.exp_config.problem.name == "tsp":
                # The runner's TSP method is not yet refactored to use ExperimentConfig fully.
                # We extract params for it here.
                algo_params = {algo.name: algo.params for algo in self.exp_config.algorithms}
                results = runner.run_tsp_comparison(
                    n_cities=self.exp_config.problem.dim,
                    max_iter=self.exp_config.problem.max_iter,
                    n_runs=self.exp_config.n_runs,
                    algo_params=algo_params,
                    progress_callback=progress_callback
                )
            else:
                # The continuous comparison method is refactored.
                results = runner.run_continuous_comparison(
                    self.exp_config,
                    progress_callback
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


class ComparisonTab:
    def __init__(self, parent):
        self.parent = parent
        self.is_running = False
        self.current_view = "convergence"
        self.metric_data = {}
        self.generated_figures = {}
        self.param_entries = {}
        self.spinner_running = False
        self.worker = None
        self.problem_seeds = {}
        
        # Setup layout
        layout = QHBoxLayout(parent)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        
        self._create_sidebar(layout)
        self._create_results_area(layout)
        self.change_experiment(self.comparison_menu.currentText())
        self._init_placeholder_text()
        
    def _create_sidebar(self, parent_layout):
        sidebar = QFrame()
        sidebar.setFixedWidth(280)
        sidebar.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 12px;
                border: 1px solid #3a3a3a;
            }
        """)
        parent_layout.addWidget(sidebar)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Experiment selection
        exp_label = QLabel("Experiment Group")
        exp_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        exp_label.setStyleSheet("color: white;")
        layout.addWidget(exp_label)
        
        self.comparison_menu = QComboBox()
        self.comparison_menu.addItems([
            "ACO vs SA (TSP)",
            "PSO vs HC (Rastrigin)",
            "ABC vs GA (Rastrigin)",
            "FA vs SA (Ackley)",
            "CS vs SA (Ackley)"
        ])
        self.comparison_menu.currentTextChanged.connect(self.change_experiment)
        self._style_combobox(self.comparison_menu)
        layout.addWidget(self.comparison_menu)
        
        # Parameters
        params_label = QLabel("Configuration")
        params_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        params_label.setStyleSheet("color: white; margin-top: 10px;")
        layout.addWidget(params_label)
        
        # Scrollable parameters area
        self.params_scroll = QScrollArea()
        self.params_scroll.setWidgetResizable(True)
        self.params_scroll.setStyleSheet("""
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #00A86B;
                border-radius: 6px;
            }
        """)
        
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background-color: #1e1e1e;")
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(8)
        self.params_scroll.setWidget(self.scroll_widget)
        
        layout.addWidget(self.params_scroll)

        # Seed input
        seed_label = QLabel("Experiment Seed")
        seed_label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        seed_label.setStyleSheet("color: white; margin-top: 8px;")
        layout.addWidget(seed_label)

        self.seed_entry = QLineEdit()
        self.seed_entry.setPlaceholderText("Leave empty for random")
        self._style_lineedit(self.seed_entry)
        layout.addWidget(self.seed_entry)
        
        # Action buttons
        self.run_button = QPushButton("▶ Run Benchmark")
        self.run_button.clicked.connect(self.run_experiment)
        self._style_button(self.run_button, "#00A86B", "#00D9A5")
        layout.addWidget(self.run_button)
        
        self.cancel_button = QPushButton("⏹ Cancel Benchmark")
        self.cancel_button.clicked.connect(self.cancel_experiment)
        self._style_button(self.cancel_button, "#FF6B6B", "#FF4D4D")
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        
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
                border-radius: 8px;
                border: 1px solid #4a4a4a;
            }
        """)
        progress_layout = QHBoxLayout(self.progress_frame)
        
        self.spinner_label = QLabel("⏳")
        self.spinner_label.setFont(QFont("Segoe UI", 18))
        progress_layout.addWidget(self.spinner_label)
        
        self.progress_message = QLabel("")
        self.progress_message.setFont(QFont("Segoe UI", 10))
        self.progress_message.setStyleSheet("color: #aaaaaa;")
        self.progress_message.setWordWrap(True)
        progress_layout.addWidget(self.progress_message, 1)
        
        self.progress_frame.hide()
        layout.addWidget(self.progress_frame)
        
        # Status
        self.status_label = QLabel("● Ready")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #aaaaaa; padding: 10px;")
        layout.addWidget(self.status_label)
        
    def _create_header(self):
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background-color: #3a3a3a;
                border-radius: 8px;
            }
        """)
        layout = QVBoxLayout(header)
        
        title = QLabel("Algorithm Comparison")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        
        subtitle = QLabel("Head-to-head performance analysis")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: #aaaaaa;")
        layout.addWidget(subtitle)
        
        return header
        
    def _style_combobox(self, combo):
        combo.setStyleSheet("""
            QComboBox {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #4a4a4a;
                border-radius: 6px;
                padding: 8px;
                font-size: 11px;
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
                padding: 6px;
                font-size: 11px;
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
                border-radius: 6px;
                padding: 10px;
                font-size: 12px;
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
        
    def _add_section_label(self, text):
        section = QLabel(text)
        section.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        section.setStyleSheet("color: #00D9A5; margin-top: 10px;")
        self.scroll_layout.addWidget(section)
        
    def _add_section_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #4a4a4a; max-height: 1px; margin: 6px 5px;")
        self.scroll_layout.addWidget(sep)
        
    def _add_param_entry(self, key, label, default):
        label_widget = QLabel(f"{label}:")
        label_widget.setFont(QFont("Segoe UI", 10))
        label_widget.setStyleSheet("color: white;")
        self.scroll_layout.addWidget(label_widget)
        
        entry = QLineEdit()
        entry.setPlaceholderText(default)
        entry.setText(default)
        self._style_lineedit(entry)
        self.scroll_layout.addWidget(entry)
        
        self.param_entries[key] = entry
        
    def change_experiment(self, experiment):
        # Clear existing widgets
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.param_entries = {}
        
        if "TSP" in experiment:
            self._add_section_label("General Settings")
            self._add_param_entry("n_cities", "Number of Cities", "20")
            self._add_param_entry("max_iter", "Maximum Iterations", "100")
            self._add_param_entry("n_runs", "Benchmark Runs", "5")
            
            self._add_section_separator()
            self._add_section_label("Ant Colony Optimization")
            self._add_param_entry("ACO_population_size", "Population Size", "20")
            self._add_param_entry("ACO_alpha", "Pheromone Weight (α)", "1.0")
            self._add_param_entry("ACO_beta", "Heuristic Weight (β)", "2.0")
            self._add_param_entry("ACO_evaporation_rate", "Evaporation Rate (ρ)", "0.1")
            
            self._add_section_separator()
            self._add_section_label("Simulated Annealing")
            self._add_param_entry("SA_initial_temp", "Initial Temperature", "1000")
            self._add_param_entry("SA_final_temp", "Final Temperature", "0.001")
            self._add_param_entry("SA_alpha", "Cooling Factor (α)", "0.995")
        else:
            self._add_section_label("General Settings")
            self._add_param_entry("dim", "Problem Dimensions", "10")
            self._add_param_entry("max_iter", "Maximum Iterations", "100")
            self._add_param_entry("n_runs", "Benchmark Runs", "5")
            
            self._add_section_separator()
            
            if "PSO vs HC" in experiment:
                self._add_section_label("Particle Swarm Optimization")
                self._add_param_entry("PSO_population_size", "Population Size", "30")
                self._add_param_entry("PSO_w", "Inertia Weight", "0.7298")
                self._add_param_entry("PSO_c1", "Cognitive (c₁)", "1.49618")
                self._add_param_entry("PSO_c2", "Social (c₂)", "1.49618")
                
                self._add_section_separator()
                self._add_section_label("Hill Climbing")
                self._add_param_entry("HC_step_size", "Step Size", "0.1")
            elif "ABC vs GA" in experiment:
                self._add_section_label("Artificial Bee Colony")
                self._add_param_entry("ABC_population_size", "Population Size", "30")
                self._add_param_entry("ABC_limit", "Scout Limit", "auto")
                
                self._add_section_separator()
                self._add_section_label("Genetic Algorithm")
                self._add_param_entry("GA_population_size", "Population Size", "50")
                self._add_param_entry("GA_crossover_rate", "Crossover Rate", "0.8")
                self._add_param_entry("GA_mutation_rate", "Mutation Rate", "0.1")
            elif "FA vs SA" in experiment:
                self._add_section_label("Firefly Algorithm")
                self._add_param_entry("FA_population_size", "Population Size", "25")
                self._add_param_entry("FA_alpha", "Randomness (α)", "0.5")
                self._add_param_entry("FA_beta0", "Attractiveness (β₀)", "1.0")
                self._add_param_entry("FA_gamma", "Absorption (γ)", "1.0")
                
                self._add_section_separator()
                self._add_section_label("Simulated Annealing")
                self._add_param_entry("SA_initial_temp", "Initial Temperature", "1000")
                self._add_param_entry("SA_alpha", "Cooling Factor (α)", "0.98")
            elif "CS vs SA" in experiment:
                self._add_section_label("Cuckoo Search")
                self._add_param_entry("CS_population_size", "Population Size", "25")
                self._add_param_entry("CS_pa", "Discovery Rate (pₐ)", "0.25")
                
                self._add_section_separator()
                self._add_section_label("Simulated Annealing")
                self._add_param_entry("SA_initial_temp", "Initial Temperature", "1000")
                self._add_param_entry("SA_alpha", "Cooling Factor (α)", "0.98")
        
        self.scroll_layout.addStretch()
        
    def _create_results_area(self, parent_layout):
        results = QFrame()
        results.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border-radius: 12px;
                border: 1px solid #3a3a3a;
            }
        """)
        parent_layout.addWidget(results, 1)
        
        layout = QVBoxLayout(results)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header = QLabel("Comparative Analysis Dashboard")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        header.setStyleSheet("color: white;")
        layout.addWidget(header)
        
        # Metric buttons
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet("background-color: #3a3a3a; border-radius: 8px;")
        buttons_layout = QHBoxLayout(buttons_frame)
        
        self.convergence_btn = QPushButton("Convergence Speed")
        self.complexity_btn = QPushButton("Complexity")
        self.robustness_btn = QPushButton("Robustness")
        self.scalability_btn = QPushButton("Scalability")
        
        for btn in [self.convergence_btn, self.complexity_btn, self.robustness_btn, self.scalability_btn]:
            self._style_metric_button(btn)
            buttons_layout.addWidget(btn)
            
        self.convergence_btn.clicked.connect(lambda: self.show_metric_view("convergence"))
        self.complexity_btn.clicked.connect(lambda: self.show_metric_view("complexity"))
        self.robustness_btn.clicked.connect(lambda: self.show_metric_view("robustness"))
        self.scalability_btn.clicked.connect(lambda: self.show_metric_view("scalability"))
        
        layout.addWidget(buttons_frame)
        
        # Content area
        self.metric_content_frame = QFrame()
        self.metric_content_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border-radius: 8px;
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
                border-radius: 6px;
                padding: 10px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                color: white;
            }
        """)
        
    def _init_placeholder_text(self):
        placeholder = "Select an experiment and click 'Run Benchmark' to start analysis."
        for view in ["convergence", "complexity", "robustness", "scalability"]:
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
        for btn in [self.convergence_btn, self.complexity_btn, self.robustness_btn, self.scalability_btn]:
            self._style_metric_button(btn)
            
        # Highlight active button
        button_map = {
            "convergence": self.convergence_btn,
            "complexity": self.complexity_btn,
            "robustness": self.robustness_btn,
            "scalability": self.scalability_btn
        }
        if view_name in button_map:
            button_map[view_name].setStyleSheet("""
                QPushButton {
                    background-color: #00A86B;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-size: 11px;
                    font-weight: bold;
                }
            """)
        
        # Display content
        # Always regenerate the figure to avoid rendering issues with reused figure objects.
        fig = None
        data = self.metric_data
        
        # Check if we have actual results data to plot
        if isinstance(data.get('metadata'), dict):
            plot_map = {
                "convergence": self._plot_convergence,
                "complexity": self._plot_complexity,
                "robustness": self._plot_robustness,
                "scalability": self._plot_scalability,
            }
            if view_name in plot_map:
                # Generate a new figure
                fig = plot_map[view_name](data)
        
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
            placeholder = self.metric_data.get(view_name, "No data available.")
            textbox = QTextEdit()
            textbox.setReadOnly(True)
            textbox.setText(str(placeholder))
            textbox.setStyleSheet("""
                QTextEdit {
                    background-color: #3a3a3a;
                    color: white;
                    border: 1px solid #4a4a4a;
                    border-radius: 4px;
                    padding: 10px;
                    font-family: Consolas;
                    font-size: 10px;
                }
            """)
            self.metric_display_layout.addWidget(textbox)
            
    def _plot_convergence(self, data):
        is_tsp = "main_results" in data
        if is_tsp:
            fig, axes = plt.subplots(1, len(data['main_results']), figsize=(7 * len(data['main_results']), 6), squeeze=False)
            axes = axes.flatten()
            for ax, (algo_name, res) in zip(axes, data['main_results'].items()):
                plot_tsp_route(data['metadata']['cities'], res['best_route'], res['best_distance'], 
                             title=f"{algo_name}\nDistance: {res['best_distance']:.2f}", ax=ax)
        else:
            fig, ax = plt.subplots(figsize=(12, 7))
            histories = {s.algorithm_name: s.results[0].history for s in data['stats_list'] 
                        if s.results and hasattr(s.results[0], 'history')}
            final_fitnesses = {s.algorithm_name: s.mean_fitness for s in data['stats_list']}
            plot_convergence_comparison(histories, 
                                      title=f"Convergence: {data['metadata']['problem'].capitalize()} (dim={data['metadata']['dim']})", 
                                      ax=ax, log_scale=True,
                                      final_fitness_values=final_fitnesses)
        plt.tight_layout()
        return fig

    def _plot_complexity(self, data):
        is_tsp = "main_results" in data
        if is_tsp:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle("TSP Computational Complexity", fontsize=16, fontweight='bold')
            
            results = data['main_results']
            algo_names = list(results.keys())
            mean_times = [r['mean_time'] for r in results.values()]
            mean_mems = [r.get('mean_mem', 0) for r in results.values()] # Use .get for safety
            colors = sns.color_palette("viridis", len(algo_names))
            
            bars1 = ax1.bar(algo_names, mean_times, color=colors, alpha=0.8)
            ax1.set_title("Mean Execution Time", fontsize=14)
            ax1.set_ylabel("Time (seconds)", fontsize=12)
            for bar in bars1:
                yval = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}s', 
                         va='bottom', ha='center', fontsize=9)

            bars2 = ax2.bar(algo_names, mean_mems, color=colors, alpha=0.8)
            ax2.set_title("Mean Memory Usage", fontsize=14)
            ax2.set_ylabel("Peak Memory (MB)", fontsize=12)
            for bar in bars2:
                yval = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f} MB', 
                         va='bottom', ha='center', fontsize=9)

            for ax in [ax1, ax2]:
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            fig = plot_complexity_comparison(data['stats_list'], 
                                           title=f"Complexity: {data['metadata']['problem'].capitalize()} (dim={data['metadata']['dim']})")
        plt.tight_layout()
        return fig

    def _plot_robustness(self, data):
        is_tsp = "main_results" in data
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if is_tsp:
            results = data['main_results']
            n_runs = self.param_entries["n_runs"].text()
            if int(n_runs or 0) < 2:
                ax.text(0.5, 0.5, 'Increase "Benchmark Runs" to 2 or more\nto see TSP robustness analysis.', 
                       ha='center', va='center', fontsize=14)
                return fig
                
            boxplot_data = {name: res['all_distances'] for name, res in results.items()}
            ax.set_title("TSP Robustness - Distance Distribution", fontsize=16, fontweight='bold')
            ax.set_ylabel("Tour Distance", fontsize=13)
            
            bp = ax.boxplot(boxplot_data.values(), tick_labels=boxplot_data.keys(), 
                          patch_artist=True, showmeans=True, meanline=True)
            colors = sns.color_palette("Set3", len(boxplot_data))
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)

            # Add annotations for medians
            medians = [np.median(d) for d in boxplot_data.values()]
            for i, median in enumerate(medians):
                ax.text(i + 1, median, f'{median:.2f}',
                        horizontalalignment='center',
                        verticalalignment='bottom',
                        fontdict={'size': 9, 'color': 'black', 'weight': 'semibold'})
                
            ax.legend([Line2D([0], [0], color='red', lw=2), Line2D([0], [0], color='blue', lw=2, ls='--')], 
                     ['Median', 'Mean'])
        else:
            boxplot_data = {s.algorithm_name: s.all_fitnesses for s in data['stats_list']}
            plot_boxplot_comparison(boxplot_data, 
                                  title=f"Robustness: {data['metadata']['problem'].capitalize()} (dim={data['metadata']['dim']})", 
                                  ax=ax)
        plt.tight_layout()
        return fig

    def _plot_scalability(self, data):
        is_tsp = "main_results" in data
        
        if is_tsp:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            fig.suptitle("TSP Scalability Analysis", fontsize=18, fontweight='bold')
            
            scal_data = data['scalability_data']
            colors = sns.color_palette("husl", len(scal_data))
            
            for (name, s_data), color in zip(scal_data.items(), colors):
                ax1.plot(s_data['cities'], s_data['distances'], marker='o', linestyle='-', color=color, label=name)
                ax2.plot(s_data['cities'], s_data['times'], marker='o', linestyle='-', color=color, label=name)
            
            ax1.set_title("Solution Quality vs. Problem Size")
            ax1.set_xlabel("Number of Cities")
            ax1.set_ylabel("Tour Distance")
            
            ax2.set_title("Execution Time vs. Problem Size")
            ax2.set_xlabel("Number of Cities")
            ax2.set_ylabel("Time (seconds)")
            
            for ax in [ax1, ax2]:
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
        else:
            fig = plot_scalability_comparison(data['scalability_data'], 
                                            title=f"Scalability: {data['metadata']['problem'].capitalize()}")
        plt.tight_layout()
        return fig
        
    def _start_spinner(self):
        self.spinner_running = True
        self.spinner_chars = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self.spinner_index = 0
        self.spinner_timer = QTimer()
        self.spinner_timer.timeout.connect(self._animate_spinner)
        self.spinner_timer.start(200)
        
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
        self.comparison_menu.setEnabled(False)
        self.reset_seed_button.setEnabled(False)
        self.seed_entry.setEnabled(False)
        for entry in self.param_entries.values():
            entry.setEnabled(False)
            
    def _enable_inputs(self):
        self.comparison_menu.setEnabled(True)
        self.reset_seed_button.setEnabled(True)
        self.seed_entry.setEnabled(True)
        for entry in self.param_entries.values():
            entry.setEnabled(True)
            
    def update_status(self, message, color="#aaaaaa"):
        icon_map = {"Ready": "●", "Running": "⏳", "Complete": "✓", "Cancelled": "⏹", "Error": "✖"}
        icon = next((icon_map[key] for key in icon_map if key in message), "●")
        self.status_label.setText(f"{icon} {message}")
        self.status_label.setStyleSheet(f"color: {color}; padding: 10px;")

    def update_progress(self, message):
        self.progress_message.setText(message)

    def _get_experiment_config(self, seed: int) -> ExperimentConfig:
        """Gathers settings from the UI and builds an ExperimentConfig object."""
        experiment = self.comparison_menu.currentText()
        raw_params = {key: entry.text() or entry.placeholderText() for key, entry in self.param_entries.items()}

        def parse_value(value_str):
            if isinstance(value_str, str) and value_str.lower() == 'auto':
                return None
            try:
                value = float(value_str)
                return int(value) if value.is_integer() else value
            except (ValueError, TypeError):
                return value_str

        # --- Problem Configuration ---
        if "TSP" in experiment:
            problem_name = "tsp"
            dim = int(raw_params.get('n_cities', 20))
        else:
            problem_map = {"Rastrigin": "rastrigin", "Ackley": "ackley"}
            problem_name = next((p for k, p in problem_map.items() if k in experiment), "rastrigin")
            dim = int(raw_params.get('dim', 10))

        problem_config = ProblemConfig(
            name=problem_name,
            dim=dim,
            max_iter=int(raw_params.get('max_iter', 100))
        )

        # --- Algorithm Configuration ---
        algo_configs = []
        algos_in_exp = []
        if "PSO vs HC" in experiment: algos_in_exp = ['PSO', 'HC']
        elif "ABC vs GA" in experiment: algos_in_exp = ['ABC', 'GA']
        elif "FA vs SA" in experiment: algos_in_exp = ['FA', 'SA']
        elif "CS vs SA" in experiment: algos_in_exp = ['CS', 'SA']
        elif "TSP" in experiment: algos_in_exp = ['ACO', 'SA']

        for algo_name in algos_in_exp:
            params = {}
            for key, value_str in raw_params.items():
                if key.startswith(f"{algo_name}_"):
                    param_name = key.split('_', 1)[1]
                    params[param_name] = parse_value(value_str)
            algo_configs.append(AlgorithmConfig(name=algo_name, params=params, enabled=True))

        # --- Experiment Configuration ---
        exp_config = ExperimentConfig(
            name=experiment,
            problem=problem_config,
            algorithms=algo_configs,
            n_runs=int(raw_params.get('n_runs', 5)),
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
        
        self.update_progress("Initializing benchmark...")
        self._start_spinner()
        self.update_status("Running...", "#FFA500")
        
        # --- Determine Seed ---
        experiment = self.comparison_menu.currentText()
        problem_key = experiment # Default key
        if "TSP" in experiment:
            n_cities_str = self.param_entries.get('n_cities').text()
            problem_key = f"tsp_{int(n_cities_str or 20)}"
        else:
            dim_str = self.param_entries.get('dim').text()
            prob_name_str = experiment.split('(')[1].split(')')[0]
            problem_key = f"{prob_name_str.lower()}_{int(dim_str or 10)}"

        seed_text = self.seed_entry.text()
        seed = None

        if seed_text.strip().isdigit():
            seed = int(seed_text)
            self.update_progress(f"Using manually entered seed: {seed}")
        else:
            if problem_key not in self.problem_seeds:
                self.problem_seeds[problem_key] = np.random.randint(0, 2**31 - 1)
            seed = self.problem_seeds[problem_key]
            self.update_progress(f"Using session seed {seed} for {problem_key}")

        # Update the UI to show the seed being used
        self.seed_entry.setText(str(seed))

        # Get the unified experiment config
        exp_config = self._get_experiment_config(seed)
        
        # Create and start worker thread
        self.worker = ComparisonWorker(exp_config, seed)
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
        experiment = self.comparison_menu.currentText()
        problem_key = experiment # Default key
        if "TSP" in experiment:
            n_cities_str = self.param_entries.get('n_cities').text()
            problem_key = f"tsp_{int(n_cities_str or 20)}"
        else:
            dim_str = self.param_entries.get('dim').text()
            prob_name_str = experiment.split('(')[1].split(')')[0]
            problem_key = f"{prob_name_str.lower()}_{int(dim_str or 10)}"

        if problem_key in self.problem_seeds:
            del self.problem_seeds[problem_key]
            self.update_status(f"Seed reset for {problem_key}. New seed will be generated.", "#00D9A5")
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
        error_text = f"✖ BENCHMARK ERROR\n\n{error_msg}"
        self.metric_data = {k: error_text for k in ["convergence", "complexity", "robustness", "scalability"]}
        self.show_metric_view(self.current_view)
        self.update_status("Error occurred", "#FF6B6B")
        
    def _experiment_cancelled(self):
        self._clear_figures()
        self.update_progress("Cancelled by user")
        self.update_status("Cancelled", "#FF6B6B")
        self.metric_data = {k: "⏹ Benchmark cancelled by user." for k in ["convergence", "complexity", "robustness", "scalability"]}
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
            self.update_status("Benchmark Complete", "#00D9A5")
        
    def save_all_figures(self):
        if not isinstance(self.metric_data.get('metadata'), dict):
            self.update_status("No results to export", "#FFD166")
            return

        experiment_name = self.comparison_menu.currentText().replace(" ", "_").replace("(", "").replace(")", "")
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir_str = f"results/figures/comparison/{experiment_name}_{timestamp}"
        output_dir = os.path.join(os.getcwd(), output_dir_str.replace('/', os.sep))
        os.makedirs(output_dir, exist_ok=True)

        views_to_save = ["convergence", "complexity", "robustness", "scalability"]
        
        plot_map = {
            "convergence": self._plot_convergence,
            "complexity": self._plot_complexity,
            "robustness": self._plot_robustness,
            "scalability": self._plot_scalability,
        }

        saved_files = []
        for view in views_to_save:
            # Always regenerate the figure to ensure it's valid and not tied to a deleted canvas
            fig = plot_map[view](self.metric_data)
            
            if fig:
                try:
                    filepath = os.path.join(output_dir, f"{view}.png")
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_files.append(os.path.basename(filepath))
                finally:
                    # Always close the figure to free up memory
                    plt.close(fig)

        if saved_files:
            self.update_status(f"Exported {len(saved_files)} plots to {output_dir_str}", "#00D9A5")
        else:
            self.update_status("No valid plots to export", "#FFD166")
