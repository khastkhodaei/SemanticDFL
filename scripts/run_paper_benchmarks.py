#!/usr/bin/env python3
"""
Automated Paper Evaluation Benchmark Suite for SemanticDFL.

Orchestrates all 5 experimental evaluation scenarios from:
Dogani, Khastkhodaei, Khunjush, Laoutaris (POMACS / ACM SIGMETRICS 2026)

Usage:
    python scripts/run_paper_benchmarks.py --scenario all --rounds 5
    python scripts/run_paper_benchmarks.py --scenario exp1_fmnist_dirichlet
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional
import yaml

from run_simulation import load_config, run_in_memory_simulation, setup_logger


# Reference baseline numbers from SemanticDFL Paper (ACM 2026)
PAPER_BASELINES: Dict[str, Dict[str, Any]] = {
    "exp1_fmnist_dirichlet": {
        "title": "Fashion-MNIST (Dirichlet alpha=0.3)",
        "dataset": "FashionMNIST",
        "model": "FashionCNN",
        "baseline_gossip_acc": 76.2,
        "semantic_dfl_target_acc": 84.8,
        "traffic_reduction_pct": 68.4,
    },
    "exp2_cifar10_pathological": {
        "title": "CIFAR-10 (Pathological 2-class)",
        "dataset": "CIFAR-10",
        "model": "ResNet18Small",
        "baseline_gossip_acc": 48.5,
        "semantic_dfl_target_acc": 67.4,
        "traffic_reduction_pct": 72.1,
    },
    "exp3_speech_commands": {
        "title": "Google Speech Commands (35 classes)",
        "dataset": "SpeechCommands",
        "model": "GoogleSpeechResNet",
        "baseline_gossip_acc": 71.0,
        "semantic_dfl_target_acc": 82.3,
        "traffic_reduction_pct": 64.7,
    },
    "exp4_ag_news_text": {
        "title": "AG News / Text NLP Classification",
        "dataset": "20Newsgroups",
        "model": "AlbertTextClassifier",
        "baseline_gossip_acc": 62.1,
        "semantic_dfl_target_acc": 78.5,
        "traffic_reduction_pct": 66.0,
    },
    "exp5_slices_wan": {
        "title": "SLICES WAN Topology (Real WAN Latencies)",
        "dataset": "FashionMNIST",
        "model": "FashionCNN",
        "baseline_gossip_acc": 74.0,
        "semantic_dfl_target_acc": 83.9,
        "traffic_reduction_pct": 70.5,
    },
}


def list_available_scenarios(scenarios_dir: str = "configs/scenarios") -> List[str]:
    """Discover all available scenario preset YAML files in the given directory."""
    if not os.path.isdir(scenarios_dir):
        return list(PAPER_BASELINES.keys())

    scenarios = []
    for f in sorted(os.listdir(scenarios_dir)):
        if f.endswith(".yaml") or f.endswith(".yml"):
            scenarios.append(os.path.splitext(f)[0])
    return scenarios or list(PAPER_BASELINES.keys())


def run_single_benchmark(
    scenario_name: str,
    rounds: Optional[int] = None,
    num_clients: Optional[int] = None,
    scenarios_dir: str = "configs/scenarios",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute a single scenario benchmark and return standardized execution metrics.
    """
    yaml_file = os.path.join(scenarios_dir, f"{scenario_name}.yaml")
    if not os.path.exists(yaml_file):
        yaml_file = os.path.join(scenarios_dir, f"{scenario_name}.yml")
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"Scenario configuration not found: {yaml_file}")

    config = load_config(yaml_file)

    # CLI parameter overrides
    if rounds is not None and rounds > 0:
        config.setdefault("simulation", {})["rounds"] = rounds
    if num_clients is not None and num_clients > 0:
        config.setdefault("federated", {})["num_clients"] = num_clients
    if device is not None:
        config.setdefault("simulation", {})["device"] = device

    # Setup logger and output directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.get("logging", {}).get("output_dir", f"results/scenarios/{scenario_name}")
    logger, _ = setup_logger(output_dir=out_dir, timestamp_str=timestamp_str)

    args = argparse.Namespace(
        config=yaml_file,
        rounds=config.get("simulation", {}).get("rounds", 10),
        num_clients=config.get("federated", {}).get("num_clients", 10),
        device=config.get("simulation", {}).get("device", "auto"),
        mode="simulation",
        backend="memory",
    )

    result = run_in_memory_simulation(
        config=config,
        args=args,
        logger=logger,
        timestamp_str=timestamp_str,
    )

    baseline_info = PAPER_BASELINES.get(scenario_name, {})
    mean_acc = result.get("summary", {}).get("mean_accuracy", 0.0)
    std_acc = result.get("summary", {}).get("std_accuracy", 0.0)
    duration = result.get("total_duration_seconds", 0.0)

    benchmark_entry = {
        "scenario": scenario_name,
        "title": baseline_info.get("title", scenario_name),
        "rounds_executed": args.rounds,
        "num_clients": args.num_clients,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "duration_seconds": duration,
        "paper_baseline_gossip_acc": baseline_info.get("baseline_gossip_acc"),
        "paper_semantic_dfl_target_acc": baseline_info.get("semantic_dfl_target_acc"),
        "paper_traffic_reduction_pct": baseline_info.get("traffic_reduction_pct"),
        "results_data": result,
    }

    return benchmark_entry


def run_all_benchmarks(
    rounds: Optional[int] = None,
    num_clients: Optional[int] = None,
    scenarios_dir: str = "configs/scenarios",
    output_path: str = "results/benchmark_summary.json",
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full suite of paper evaluation benchmarks and aggregate summary metrics.
    """
    scenarios = list_available_scenarios(scenarios_dir)
    print(f"\n[SemanticDFL Benchmark Suite] Found {len(scenarios)} scenarios to execute.")
    print(f"Scenarios: {', '.join(scenarios)}\n")

    suite_start = datetime.now()
    results: Dict[str, Any] = {}

    for idx, sc in enumerate(scenarios, start=1):
        print(f"\n>>> Running [{idx}/{len(scenarios)}] {sc}...")
        try:
            entry = run_single_benchmark(
                scenario_name=sc,
                rounds=rounds,
                num_clients=num_clients,
                scenarios_dir=scenarios_dir,
                device=device,
            )
            results[sc] = entry
        except Exception as e:
            print(f"Error executing benchmark {sc}: {e}", file=sys.stderr)
            results[sc] = {"scenario": sc, "status": "FAILED", "error": str(e)}

    suite_duration = (datetime.now() - suite_start).total_seconds()

    summary_payload = {
        "timestamp": suite_start.isoformat(),
        "total_suite_duration_seconds": suite_duration,
        "num_scenarios": len(scenarios),
        "scenarios": results,
    }

    # Save summary JSON
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n[SemanticDFL Benchmark Suite] All benchmarks completed in {suite_duration:.2f}s.")
    print(f"Summary saved to -> {output_path}")

    print_summary_table(summary_payload)
    return summary_payload


def print_summary_table(summary: Dict[str, Any]) -> None:
    """Print an academic comparative performance table to stdout."""
    print("\n" + "=" * 95)
    print("                SemanticDFL (ACM 2026) Paper Benchmark Suite Summary")
    print("=" * 95)
    print(
        f"{'Scenario':<28} | {'Clients':<7} | {'Rounds':<6} | {'Achieved Acc':<13} | "
        f"{'Paper Target':<12} | {'Baseline Acc':<12}"
    )
    print("-" * 95)

    scenarios = summary.get("scenarios", {})
    for sc_name, sc_data in scenarios.items():
        if sc_data.get("status") == "FAILED":
            print(f"{sc_name:<28} | {'FAILED':<7} | {'-':<6} | {'-':<13} | {'-':<12} | {'-':<12}")
            continue

        clients = sc_data.get("num_clients", "-")
        rnds = sc_data.get("rounds_executed", "-")
        acc = sc_data.get("mean_accuracy", 0.0)
        std = sc_data.get("std_accuracy", 0.0)
        target = sc_data.get("paper_semantic_dfl_target_acc", "-")
        baseline = sc_data.get("paper_baseline_gossip_acc", "-")

        acc_str = f"{acc:.2f}% ± {std:.1f}" if isinstance(acc, (int, float)) else str(acc)
        target_str = f"{target:.1f}%" if isinstance(target, (int, float)) else str(target)
        baseline_str = f"{baseline:.1f}%" if isinstance(baseline, (int, float)) else str(baseline)

        print(
            f"{sc_name:<28} | {clients:<7} | {rnds:<6} | {acc_str:<13} | "
            f"{target_str:<12} | {baseline_str:<12}"
        )

    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run SemanticDFL Paper Evaluation Benchmarks and Comparison Suite"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Scenario to run: 'all' or specific scenario name (e.g. 'exp1_fmnist_dirichlet')",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Override total communication rounds across scenarios",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Override number of clients (useful for fast verification runs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Hardware execution device ('cpu', 'cuda', or 'auto')",
    )
    parser.add_argument(
        "--scenarios_dir",
        type=str,
        default="configs/scenarios",
        help="Directory containing scenario preset YAML files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_summary.json",
        help="Target output JSON path for the summary metrics",
    )

    args = parser.parse_args()

    if args.scenario == "all":
        run_all_benchmarks(
            rounds=args.rounds,
            num_clients=args.num_clients,
            scenarios_dir=args.scenarios_dir,
            output_path=args.output,
            device=args.device,
        )
    else:
        entry = run_single_benchmark(
            scenario_name=args.scenario,
            rounds=args.rounds,
            num_clients=args.num_clients,
            scenarios_dir=args.scenarios_dir,
            device=args.device,
        )
        single_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_suite_duration_seconds": entry.get("duration_seconds", 0.0),
            "num_scenarios": 1,
            "scenarios": {args.scenario: entry},
        }
        print_summary_table(single_summary)


if __name__ == "__main__":
    main()
