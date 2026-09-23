"""
SemanticDFL Visualization and Plotting Package.
"""

from semantic_dfl.visualization.paper_plots import (
    generate_paper_evaluation_plots,
    plot_figure_1_accuracy_curves,
    plot_figure_2_search_recall,
    plot_figure_3_traffic_breakdown,
    plot_figure_4_adaptive_threshold,
    plot_figure_5_personalization_coupling,
    plot_figure_6_freshness_guard,
    plot_figure_7_topology_sensitivity,
    set_paper_academic_style,
)
from semantic_dfl.visualization.plotter import (
    generate_all_plots,
    load_run_data,
    plot_accuracy_curve,
    plot_loss_curve,
    plot_recall_and_cardinality,
    plot_time_breakdown,
    set_academic_style,
)

__all__ = [
    "load_run_data",
    "plot_accuracy_curve",
    "plot_loss_curve",
    "plot_recall_and_cardinality",
    "plot_time_breakdown",
    "generate_all_plots",
    "set_academic_style",
    "generate_paper_evaluation_plots",
    "plot_figure_1_accuracy_curves",
    "plot_figure_2_search_recall",
    "plot_figure_3_traffic_breakdown",
    "plot_figure_4_adaptive_threshold",
    "plot_figure_5_personalization_coupling",
    "plot_figure_6_freshness_guard",
    "plot_figure_7_topology_sensitivity",
    "set_paper_academic_style",
]

