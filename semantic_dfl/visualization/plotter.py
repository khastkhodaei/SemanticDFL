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
