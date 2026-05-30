"""Command-line entry point for solving and visualizing one benchmark instance.

Example:
    python scripts/run_single_instance.py --instance data/benchmarks/benchmark_01.json --algorithm ga

The script:
- loads one JSON benchmark instance,
- executes the selected optimization algorithm,
- prints metrics in the terminal,
- exports metrics and convergence CSV files,
- saves a route visualization when matplotlib is installed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_delivery.algorithms.exact import (  # noqa: E402
    BranchAndBoundConfig,
    BranchAndBoundResult,
    solve_branch_and_bound,
)
from drone_delivery.algorithms.metaheuristics import (  # noqa: E402
    GeneticAlgorithmConfig,
    GeneticAlgorithmResult,
    SimulatedAnnealingConfig,
    SimulatedAnnealingResult,
    solve_genetic_algorithm,
    solve_simulated_annealing,
)
from drone_delivery.core import ProblemInstance  # noqa: E402
from drone_delivery.core.solution import Solution  # noqa: E402
from drone_delivery.data.io import load_instance  # noqa: E402
from drone_delivery.evaluation.metrics import (  # noqa: E402
    AlgorithmMetrics,
    ConvergencePoint,
    evaluate_solution_feasibility,
    export_convergence_to_csv,
    export_metrics_to_csv,
    solution_distance,
    solution_objective,
)


def parse_args() -> argparse.Namespace:
    """Parse terminal arguments."""

    parser = argparse.ArgumentParser(
        description="Solve one drone delivery benchmark instance.",
    )
    parser.add_argument(
        "--instance",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks" / "benchmark_01.json",
        help="Path to one benchmark JSON file.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("bnb", "ga", "sa"),
        default="ga",
        help="Optimization method: bnb, ga, or sa.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "tables",
        help="Directory where CSV metrics are saved.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "plots",
        help="Directory where route plots are saved.",
    )
    parser.add_argument(
        "--bnb-max-nodes",
        type=int,
        default=20_000,
        help="Maximum B&B nodes. Use 0 for no node limit.",
    )
    parser.add_argument(
        "--bnb-time-limit",
        type=float,
        default=5.0,
        help="B&B time limit in seconds. Use 0 for no time limit.",
    )
    parser.add_argument(
        "--ga-population",
        type=int,
        default=60,
        help="GA population size.",
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
        help="Random seed for GA/SA.",
    )
    parser.add_argument(
        "--base-energy-rate",
        type=float,
        default=1.0,
        help="Base energy per distance unit.",
    )
    parser.add_argument(
        "--payload-energy-rate",
        type=float,
        default=0.03,
        help="Additional energy per distance and payload unit.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip PNG generation. Useful if matplotlib is not installed.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the selected algorithm and save outputs."""

    args = parse_args()
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    instance = ProblemInstance.from_dict(load_instance(args.instance))
    print(f"Loaded instance: {instance.instance_id}")
    print(f"Customers: {len(instance.customers)}")
    print(f"Drones: {len(instance.drones)}")
    print(f"No-fly zones: {len(instance.no_fly_zones)}")
    print(f"Algorithm: {args.algorithm}\n")

    result = _run_algorithm(instance, args)
    metrics = _build_metrics(instance, args.algorithm, result, args)
    convergence = _build_convergence(instance.instance_id, args.algorithm, result)

    _print_metrics(metrics)

    metrics_path = export_metrics_to_csv(
        [metrics],
        args.tables_dir / f"{instance.instance_id}_{args.algorithm}_metrics.csv",
    )
    convergence_path = export_convergence_to_csv(
        convergence,
        args.tables_dir / f"{instance.instance_id}_{args.algorithm}_convergence.csv",
    )
    print(f"\nSaved metrics CSV: {metrics_path}")
    print(f"Saved convergence CSV: {convergence_path}")

    if args.skip_plot:
        print("Skipped route visualization.")
        return

    plot_path = _save_visualization(instance, _extract_solution(result), args)
    if plot_path is not None:
        print(f"Saved plot: {plot_path}")


def _run_algorithm(
    instance: ProblemInstance,
    args: argparse.Namespace,
) -> BranchAndBoundResult | GeneticAlgorithmResult | SimulatedAnnealingResult:
    """Run the selected optimization algorithm."""

    if args.algorithm == "bnb":
        return solve_branch_and_bound(
            instance,
            BranchAndBoundConfig(
                base_energy_rate=args.base_energy_rate,
                payload_energy_rate=args.payload_energy_rate,
                max_nodes=None if args.bnb_max_nodes == 0 else args.bnb_max_nodes,
                time_limit_seconds=None if args.bnb_time_limit == 0 else args.bnb_time_limit,
            ),
        )

    if args.algorithm == "ga":
        return solve_genetic_algorithm(
            instance,
            GeneticAlgorithmConfig(
                population_size=args.ga_population,
                generations=args.ga_generations,
                random_seed=args.seed,
                base_energy_rate=args.base_energy_rate,
                payload_energy_rate=args.payload_energy_rate,
            ),
        )

    return solve_simulated_annealing(
        instance,
        SimulatedAnnealingConfig(
            initial_temperature=args.sa_initial_temperature,
            max_iterations=args.sa_iterations,
            random_seed=args.seed,
            base_energy_rate=args.base_energy_rate,
            payload_energy_rate=args.payload_energy_rate,
        ),
    )


def _build_metrics(
    instance: ProblemInstance,
    algorithm_code: str,
    result: BranchAndBoundResult | GeneticAlgorithmResult | SimulatedAnnealingResult,
    args: argparse.Namespace,
) -> AlgorithmMetrics:
    """Convert one algorithm result to a common metric row."""

    solution = _extract_solution(result)
    feasibility = evaluate_solution_feasibility(
        solution,
        base_energy_rate=args.base_energy_rate,
        payload_energy_rate=args.payload_energy_rate,
    )

    if isinstance(result, BranchAndBoundResult):
        penalized_cost = solution_objective(
            solution,
            args.base_energy_rate,
            args.payload_energy_rate,
        )
        return AlgorithmMetrics(
            instance_id=instance.instance_id,
            algorithm="Branch and Bound",
            status=result.status,
            feasible=feasibility.feasible,
            objective_value=penalized_cost if feasibility.feasible else math.inf,
            penalized_cost=penalized_cost,
            runtime_seconds=result.runtime_seconds,
            total_distance=solution_distance(solution),
            explored_nodes=result.explored_nodes,
            pruned_by_bound=result.pruned_by_bound,
            pruned_by_infeasibility=result.pruned_by_infeasibility,
            completed_solutions=result.completed_solutions,
            feasibility_messages=" | ".join(feasibility.messages),
        )

    if isinstance(result, GeneticAlgorithmResult):
        return AlgorithmMetrics(
            instance_id=instance.instance_id,
            algorithm="Genetic Algorithm",
            status=result.status,
            feasible=feasibility.feasible,
            objective_value=result.best_objective if feasibility.feasible else math.inf,
            penalized_cost=result.best_penalized_cost,
            runtime_seconds=result.runtime_seconds,
            total_distance=solution_distance(solution),
            generations=result.generations_executed,
            feasibility_messages=" | ".join(feasibility.messages),
        )

    return AlgorithmMetrics(
        instance_id=instance.instance_id,
        algorithm="Simulated Annealing",
        status=result.status,
        feasible=feasibility.feasible,
        objective_value=result.best_objective if feasibility.feasible else math.inf,
        penalized_cost=result.best_penalized_cost,
        runtime_seconds=result.runtime_seconds,
        total_distance=solution_distance(solution),
        iterations=result.iterations_executed,
        feasibility_messages=" | ".join(feasibility.messages),
    )


def _build_convergence(
    instance_id: str,
    algorithm_code: str,
    result: BranchAndBoundResult | GeneticAlgorithmResult | SimulatedAnnealingResult,
) -> list[ConvergencePoint]:
    """Convert solver-specific logs into convergence rows."""

    if isinstance(result, BranchAndBoundResult):
        return [
            ConvergencePoint(
                instance_id=instance_id,
                algorithm="Branch and Bound",
                step=log.explored_nodes,
                best_objective=log.best_objective,
                best_penalized_cost=log.best_objective,
                current_cost=log.current_objective,
                lower_bound=log.lower_bound,
                event=log.event,
            )
            for log in result.logs
        ]

    if isinstance(result, GeneticAlgorithmResult):
        return [
            ConvergencePoint(
                instance_id=instance_id,
                algorithm="Genetic Algorithm",
                step=log.generation,
                best_objective=log.best_objective,
                best_penalized_cost=log.best_penalized_cost,
                current_cost=log.average_penalized_cost,
                event=f"feasible_count={log.feasible_count}",
            )
            for log in result.logs
        ]

    return [
        ConvergencePoint(
            instance_id=instance_id,
            algorithm="Simulated Annealing",
            step=log.iteration,
            best_objective=log.best_objective,
            best_penalized_cost=log.best_penalized_cost,
            current_cost=log.current_cost,
            temperature=log.temperature,
            event=log.event,
        )
        for log in result.logs
    ]


def _extract_solution(
    result: BranchAndBoundResult | GeneticAlgorithmResult | SimulatedAnnealingResult,
) -> Solution | None:
    """Return a solution object from any solver result."""

    return result.solution


def _save_visualization(
    instance: ProblemInstance,
    solution: Solution | None,
    args: argparse.Namespace,
) -> Path | None:
    """Save route plot if possible; fall back to instance plot if no solution exists."""

    try:
        from drone_delivery.visualization import plot_instance, plot_solution
    except ModuleNotFoundError as exc:
        if exc.name == "matplotlib":
            print(
                "Matplotlib is not installed, so visualization was not generated. "
                "Install dependencies with: pip install -e ."
            )
            return None
        raise

    if solution is None:
        return plot_instance(
            instance,
            output_dir=args.plots_dir,
            file_name=f"{instance.instance_id}_{args.algorithm}_instance.png",
        )

    return plot_solution(
        solution,
        output_dir=args.plots_dir,
        file_name=f"{instance.instance_id}_{args.algorithm}_routes.png",
        base_energy_rate=args.base_energy_rate,
        payload_energy_rate=args.payload_energy_rate,
    )


def _print_metrics(metrics: AlgorithmMetrics) -> None:
    """Display a compact terminal summary."""

    print("Result summary")
    print("--------------")
    print(f"Status: {metrics.status}")
    print(f"Feasible: {metrics.feasible}")
    print(f"Objective value: {_format_float(metrics.objective_value)}")
    print(f"Penalized cost: {_format_float(metrics.penalized_cost)}")
    print(f"Total distance: {_format_float(metrics.total_distance)}")
    print(f"Runtime: {metrics.runtime_seconds:.4f} seconds")

    if metrics.explored_nodes is not None:
        print(f"Explored nodes: {metrics.explored_nodes}")
    if metrics.generations is not None:
        print(f"Generations: {metrics.generations}")
    if metrics.iterations is not None:
        print(f"Iterations: {metrics.iterations}")
    if metrics.feasibility_messages:
        print(f"Feasibility notes: {metrics.feasibility_messages}")


def _format_float(value: float) -> str:
    """Format finite floats and hide infinities cleanly."""

    if isinstance(value, float) and not math.isfinite(value):
        return "N/A"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
