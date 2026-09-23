"""
Structured Logging, Node Log Sinks, and Execution Telemetry for SemanticDFL.

Provides:
- Per-node dedicated file logger sinks: logs/nodes/node_{id}.log
- Global simulation execution logger
- Per-round node role state JSON ledger: results/node_roles_round_{r}.json
- Full node lifecycle history ledger: results/node_lifecycle_history.json
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def setup_node_logger(
    node_id: int,
    log_dir: str = "logs/nodes",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures a dedicated file-sink logger for a specific peer node.
    
    Logs to: {log_dir}/node_{node_id}.log
    """
    os.makedirs(log_dir, exist_ok=True)
    logger_name = f"node_{node_id}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_file = os.path.join(log_dir, f"node_{node_id}.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [Node-%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_simulation_logger(
    output_dir: str = "logs",
    timestamp: Optional[str] = None,
    level: int = logging.INFO,
) -> Tuple[logging.Logger, str]:
    """
    Configures global simulation logger writing to stdout and persistent file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(output_dir, f"simulation_{ts}.log")

    logger = logging.getLogger("SemanticDFL_Global")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger, log_filepath


def save_round_node_roles(
    round_num: int,
    node_states: Dict[int, Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """
    Saves structured per-node role and telemetry state for a communication round.
    
    File path: {output_dir}/node_roles_round_{round_num}.json
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"node_roles_round_{round_num}.json")

    # Serialize node_states converting int keys to strings for JSON
    serializable = {
        "round": round_num,
        "timestamp": datetime.now().isoformat(),
        "total_nodes": len(node_states),
        "nodes": {str(k): v for k, v in node_states.items()},
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return filepath


def save_node_lifecycle_history(
    history: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """
    Saves full node lifecycle ledger across all completed rounds.
    
    File path: {output_dir}/node_lifecycle_history.json
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "node_lifecycle_history.json")

    serializable = {
        "total_rounds": len(history),
        "updated_at": datetime.now().isoformat(),
        "history": history,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    return filepath
