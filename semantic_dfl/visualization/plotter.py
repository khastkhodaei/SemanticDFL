"""
Visualization and Academic Figure Generation Engine for SemanticDFL.

Parses simulation results JSON outputs and generates publication-quality plots:
1. Accuracy convergence trajectory with client spread.
2. Training and test loss progression.
3. Dual-axis Recall@K and signature cardinality P dynamics.
4. Stacked breakdown of wall-clock phase execution times.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt


def load_run_data(filepath: str) -> Dict[str, Any]:
    """
    Parses a SemanticDFL simulation results JSON file.
    
    Args:
        filepath: Path to simulation_results_*.json or latest_results.json.
        
    Returns:
        Dictionary containing simulation config, summary, per-client metrics, and round_history.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Simulation results file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "round_history" not in data:
        raise ValueError(f"Invalid results format: missing 'round_history' in {filepath}")

    return data


def plot_accuracy_curve(
    data: Dict[str, Any],
    output_path: Optional[str] = None,
    title: str = "Test Accuracy Convergence Trajectory",
) -> plt.Figure:
    """
    Plots the average test accuracy convergence curve across federated rounds.
    
    Args:
        data: Parsed simulation results dictionary.
        output_path: Optional destination path to save the generated figure.
        title: Plot title.
        
    Returns:
        Matplotlib Figure object.
    """
    history = data["round_history"]
    rounds = [h["round"] for h in history]
    accs = [h["avg_test_accuracy"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, accs, marker="o", color="#1f77b4", linewidth=2.0, label="Mean Test Accuracy")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig
