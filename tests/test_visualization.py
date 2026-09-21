"""
Unit and Integration Tests for SemanticDFL Visualization and Plotting Engine.
"""

import os
import json
import tempfile
import pytest
import matplotlib.pyplot as plt

from semantic_dfl.visualization.plotter import (
    generate_all_plots,
    load_run_data,
    plot_accuracy_curve,
    plot_loss_curve,
    plot_recall_and_cardinality,
    plot_time_breakdown,
)


@pytest.fixture
def sample_results_data():
    return {
        "timestamp": "20260921_120000",
        "config": {
            "simulation": {"rounds": 5, "seed": 42},
            "federated": {"num_clients": 5},
        },
        "summary": {
            "mean_accuracy": 75.5,
            "std_accuracy": 3.2,
        },
        "round_history": [
            {
                "round": 1,
                "avg_test_accuracy": 35.0,
                "avg_train_loss": 1.20,
                "avg_test_loss": 1.85,
                "avg_recall_at_k": 0.70,
                "cardinality_p": 400000,
                "timings": {
                    "time_local_training": 10.0,
                    "time_signatures": 1.0,
                    "time_topology": 0.5,
                    "time_search": 0.1,
                    "time_aggregation": 0.1,
                    "time_evaluation": 4.0,
                },
            },
            {
                "round": 2,
                "avg_test_accuracy": 55.0,
                "avg_train_loss": 0.85,
                "avg_test_loss": 1.40,
                "avg_recall_at_k": 0.80,
                "cardinality_p": 320000,
                "timings": {
                    "time_local_training": 9.8,
                    "time_signatures": 0.9,
                    "time_topology": 0.4,
                    "time_search": 0.1,
                    "time_aggregation": 0.1,
                    "time_evaluation": 3.8,
                },
            },
            {
                "round": 3,
                "avg_test_accuracy": 68.0,
                "avg_train_loss": 0.60,
                "avg_test_loss": 1.05,
                "avg_recall_at_k": 0.85,
                "cardinality_p": 270000,
                "timings": {
                    "time_local_training": 9.5,
                    "time_signatures": 0.8,
                    "time_topology": 0.4,
                    "time_search": 0.1,
                    "time_aggregation": 0.1,
                    "time_evaluation": 3.7,
                },
            },
        ],
    }


def test_load_run_data_valid(sample_results_data):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sample_results_data, f)
        temp_path = f.name

    try:
        loaded = load_run_data(temp_path)
        assert len(loaded["round_history"]) == 3
        assert loaded["summary"]["mean_accuracy"] == 75.5
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_generate_all_plots(sample_results_data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        figs = generate_all_plots(sample_results_data, output_dir=tmp_dir, export_pdf=True, dpi=100)
        assert "accuracy_convergence" in figs
        assert "loss_trajectories" in figs
        assert "search_dynamics" in figs
        assert "time_breakdown" in figs

        assert os.path.exists(os.path.join(tmp_dir, "accuracy_convergence.png"))
        assert os.path.exists(os.path.join(tmp_dir, "accuracy_convergence.pdf"))
        assert os.path.exists(os.path.join(tmp_dir, "loss_trajectories.png"))
        assert os.path.exists(os.path.join(tmp_dir, "search_dynamics.png"))
        assert os.path.exists(os.path.join(tmp_dir, "time_breakdown.png"))

        plt.close("all")
