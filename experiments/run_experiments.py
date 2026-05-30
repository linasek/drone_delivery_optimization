"""Run benchmark experiments comparing Branch and Bound, GA, and SA.

This script is the main experimental pipeline. It loads benchmark JSON files,
runs all implemented algorithms, exports comparison tables, and saves
comparison plots automatically.

Example:
    python experiments/run_experiments.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_delivery.algorithms.exact import BranchAndBoundConfig
from drone_delivery.algorithms.metaheuristics import (
    GeneticAlgorithmConfig,
    SimulatedAnnealingConfig,
)
from drone_delivery.evaluation import EvaluationConfig, compare_benchmark_files


def parse_args() -> argparse.Namespace:
    """Parse command-line experiment options."""

    parser = argparse.ArgumentParser(
        description="Run drone delivery optimization experiments.",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks",
        help="Directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--pattern",
        default="benchmark_*.json",
        help="Glob pattern used to select benchmark files.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables",
        help="Directory for CSV result tables.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "plots",
        help="Directory for PNG comparison plots.",
    )
    parser.add_argument(
        "--bnb-max-nodes",
        type=int,
        default=20_000,
        help="Maximum B&B nodes per instance. Use 0 for no node limit.",
    )
    parser.add_argument(
        "--bnb-time-limit",
        type=float,
        default=5.0,
        help="B&B time limit in seconds per instance. Use 0 for no time limit.",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=60,
        help="Genetic Algorithm population size.",
    )
    parser.add_argument(
        "--ga-generations",
        type=int,
        default=150,
        help="Maximum GA generations.",
    )
    parser.add_argument(
        "--sa-iterations",
        type=int,
        default=5_000,
        help="Maximum SA iterations.",
    )
    parser.add_argument(
        "--sa-initial-temperature",
        type=float,
        default=500.0,
        help="Initial SA temperature.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for metaheuristics.",
    )
    parser.add_argument(
        "--max-convergence-plots",
        type=int,
        default=10,
        help="Maximum number of per-instance convergence plots to save.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Export tables only and skip matplotlib plot generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full experiment pipeline."""

    args = parse_args()
    benchmark_paths = sorted(args.benchmark_dir.glob(args.pattern))
    if not benchmark_paths:
        raise FileNotFoundError(
            f"No benchmark files found in {args.benchmark_dir} with pattern {args.pattern}."
        )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    evaluation_config = EvaluationConfig(
        branch_and_bound_config=BranchAndBoundConfig(
            max_nodes=None if args.bnb_max_nodes == 0 else args.bnb_max_nodes,
            time_limit_seconds=None if args.bnb_time_limit == 0 else args.bnb_time_limit,
        ),
        genetic_algorithm_config=GeneticAlgorithmConfig(
            population_size=args.ga_population,
            generations=args.ga_generations,
            random_seed=args.seed,
        ),
        simulated_annealing_config=SimulatedAnnealingConfig(
            initial_temperature=args.sa_initial_temperature,
            max_iterations=args.sa_iterations,
            random_seed=args.seed + 1,
        ),
    )

    print(f"Loaded {len(benchmark_paths)} benchmark instances.")
    for path in benchmark_paths:
        print(f"- {path.name}")

    report = compare_benchmark_files(benchmark_paths, evaluation_config)
    metrics_path, convergence_path = report.export(
        output_dir=args.tables_dir,
        metrics_file_name="comparison_metrics.csv",
        convergence_file_name="convergence_history.csv",
    )

    print(f"\nSaved metrics table: {metrics_path}")
    print(f"Saved convergence table: {convergence_path}")

    if args.skip_plots:
        print("Skipped plot generation.")
        return

    try:
        from drone_delivery.visualization.plots import generate_comparison_plots
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print(
                "\nMatplotlib is not installed, so plots were not generated. "
                "Install dependencies with: pip install -e ."
            )
            return
        raise

    plot_paths = generate_comparison_plots(
        metrics=list(report.metrics),
        convergence=list(report.convergence),
        output_dir=args.plots_dir,
        max_convergence_instances=args.max_convergence_plots,
    )

    print("\nSaved plots:")
    for path in plot_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
