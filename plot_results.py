"""
Standalone CLI Entrypoint for SemanticDFL Academic Figure Generation.

Parses simulation results JSON files and generates high-resolution academic figures
(accuracy trajectory, loss curves, search recall dynamics, and execution time breakdown).
"""

from __future__ import annotations

import argparse
import sys
import os

from semantic_dfl.visualization.plotter import generate_all_plots, load_run_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate Publication-Quality Figures for SemanticDFL Simulations"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/latest_results.json",
        help="Path to simulation results JSON file (default: results/latest_results.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to save generated figures (default: results/plots)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Resolution for PNG raster exports (default: 300)",
    )
    parser.add_argument(
        "--no_pdf",
        action="store_true",
        help="Disable vector PDF export alongside PNG figures",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("        SemanticDFL Figure Generation Engine")
    print("=" * 60)
    print(f"Input Results File: {args.input}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Export PDF:         {not args.no_pdf}")
    print(f"DPI:                {args.dpi}")
    print("=" * 60)

    try:
        data = load_run_data(args.input)
    except Exception as e:
        print(f"\n[Error] Failed to load results JSON: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nGenerating academic publication figures...")
    figures = generate_all_plots(
        data=data,
        output_dir=args.output_dir,
        export_pdf=not args.no_pdf,
        dpi=args.dpi,
    )

    print(f"\nSuccessfully rendered and saved {len(figures)} figures to '{args.output_dir}':")
    for name in figures:
        print(f"  - {name}.png / {name}.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
