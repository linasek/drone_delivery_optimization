"""Compare exact, GA, and SA results across benchmark instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from drone_delivery.algorithms.exact import (
    BranchAndBoundConfig,
    BranchAndBoundResult,
    solve_branch_and_bound,
)
from drone_delivery.algorithms.metaheuristics import (
    GeneticAlgorithmConfig,
    GeneticAlgorithmResult,
    SimulatedAnnealingConfig,
    SimulatedAnnealingResult,
    solve_genetic_algorithm,
    solve_simulated_annealing,
)
from drone_delivery.core.entities import ProblemInstance
from drone_delivery.data.io import load_instance
from drone_delivery.evaluation.metrics import (
    AlgorithmMetrics,
    ConvergencePoint,
    evaluate_solution_feasibility,
    export_convergence_to_csv,
    export_metrics_to_csv,
    solution_distance,
    solution_objective,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for comparing all implemented algorithms."""

    branch_and_bound_config: BranchAndBoundConfig = field(
        default_factory=lambda: BranchAndBoundConfig(max_nodes=50_000)
    )
    genetic_algorithm_config: GeneticAlgorithmConfig = field(
        default_factory=GeneticAlgorithmConfig
    )
    simulated_annealing_config: SimulatedAnnealingConfig = field(
        default_factory=SimulatedAnnealingConfig
    )
    run_branch_and_bound: bool = True
    run_genetic_algorithm: bool = True
    run_simulated_annealing: bool = True
    base_energy_rate: float = 1.0
    payload_energy_rate: float = 0.03
    enforce_no_fly_zones: bool = True


@dataclass(frozen=True)
class EvaluationReport:
    """Full comparison output for one or more instances."""

    metrics: tuple[AlgorithmMetrics, ...]
    convergence: tuple[ConvergencePoint, ...]

    def comparison_table(self) -> list[dict[str, object]]:
        """Return metrics as a list of dictionaries for display or CSV export."""

        return [metric.to_csv_row() for metric in self.metrics]

    def export(
        self,
        output_dir: str | Path = Path("outputs") / "tables",
        metrics_file_name: str = "comparison_metrics.csv",
        convergence_file_name: str = "convergence_history.csv",
    ) -> tuple[Path, Path]:
        """Export metrics and convergence tables to CSV files."""

        output_path = Path(output_dir)
        metrics_path = export_metrics_to_csv(
            list(self.metrics),
            output_path / metrics_file_name,
        )
        convergence_path = export_convergence_to_csv(
            list(self.convergence),
            output_path / convergence_file_name,
        )
        return metrics_path, convergence_path


def compare_algorithms(
    instance: ProblemInstance,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Run Branch and Bound, GA, and SA on one problem instance."""

    evaluation_config = config or EvaluationConfig()
    metrics: list[AlgorithmMetrics] = []
    convergence: list[ConvergencePoint] = []

    if evaluation_config.run_branch_and_bound:
        result = solve_branch_and_bound(
            instance,
            evaluation_config.branch_and_bound_config,
        )
        metrics.append(
            _metrics_from_branch_and_bound(instance, result, evaluation_config)
        )
        convergence.extend(_convergence_from_branch_and_bound(instance.instance_id, result))

    if evaluation_config.run_genetic_algorithm:
        result = solve_genetic_algorithm(
            instance,
            evaluation_config.genetic_algorithm_config,
        )
        metrics.append(_metrics_from_ga(instance, result, evaluation_config))
        convergence.extend(_convergence_from_ga(instance.instance_id, result))

    if evaluation_config.run_simulated_annealing:
        result = solve_simulated_annealing(
            instance,
            evaluation_config.simulated_annealing_config,
        )
        metrics.append(_metrics_from_sa(instance, result, evaluation_config))
        convergence.extend(_convergence_from_sa(instance.instance_id, result))

    return EvaluationReport(metrics=tuple(metrics), convergence=tuple(convergence))


def compare_benchmark_files(
    instance_paths: Iterable[str | Path],
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Run the comparison framework on multiple JSON benchmark files."""

    all_metrics: list[AlgorithmMetrics] = []
    all_convergence: list[ConvergencePoint] = []
    evaluation_config = config or EvaluationConfig()

    for path in instance_paths:
        instance = ProblemInstance.from_dict(load_instance(path))
        report = compare_algorithms(instance, evaluation_config)
        all_metrics.extend(report.metrics)
        all_convergence.extend(report.convergence)

    return EvaluationReport(
        metrics=tuple(all_metrics),
        convergence=tuple(all_convergence),
    )


def compare_benchmark_directory(
    benchmark_dir: str | Path = Path("data") / "benchmarks",
    pattern: str = "benchmark_*.json",
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Run comparison on all benchmark JSON files in a directory."""

    paths = sorted(Path(benchmark_dir).glob(pattern))
    return compare_benchmark_files(paths, config)


def _metrics_from_branch_and_bound(
    instance: ProblemInstance,
    result: BranchAndBoundResult,
    config: EvaluationConfig,
) -> AlgorithmMetrics:
    """Normalize Branch and Bound output to the common metrics schema."""

    feasibility = evaluate_solution_feasibility(
        result.solution,
        config.base_energy_rate,
        config.payload_energy_rate,
        config.enforce_no_fly_zones,
    )
    objective = solution_objective(
        result.solution,
        config.base_energy_rate,
        config.payload_energy_rate,
    )

    return AlgorithmMetrics(
        instance_id=instance.instance_id,
        algorithm="Branch and Bound",
        status=result.status,
        feasible=feasibility.feasible,
        objective_value=objective if feasibility.feasible else float("inf"),
        penalized_cost=objective,
        runtime_seconds=result.runtime_seconds,
        total_distance=solution_distance(result.solution),
        explored_nodes=result.explored_nodes,
        pruned_by_bound=result.pruned_by_bound,
        pruned_by_infeasibility=result.pruned_by_infeasibility,
        completed_solutions=result.completed_solutions,
        feasibility_messages=" | ".join(feasibility.messages),
    )


def _metrics_from_ga(
    instance: ProblemInstance,
    result: GeneticAlgorithmResult,
    config: EvaluationConfig,
) -> AlgorithmMetrics:
    """Normalize Genetic Algorithm output to the common metrics schema."""

    feasibility = evaluate_solution_feasibility(
        result.solution,
        config.base_energy_rate,
        config.payload_energy_rate,
        config.enforce_no_fly_zones,
    )

    return AlgorithmMetrics(
        instance_id=instance.instance_id,
        algorithm="Genetic Algorithm",
        status=result.status,
        feasible=feasibility.feasible,
        objective_value=result.best_objective if feasibility.feasible else float("inf"),
        penalized_cost=result.best_penalized_cost,
        runtime_seconds=result.runtime_seconds,
        total_distance=solution_distance(result.solution),
        generations=result.generations_executed,
        feasibility_messages=" | ".join(feasibility.messages),
    )


def _metrics_from_sa(
    instance: ProblemInstance,
    result: SimulatedAnnealingResult,
    config: EvaluationConfig,
) -> AlgorithmMetrics:
    """Normalize Simulated Annealing output to the common metrics schema."""

    feasibility = evaluate_solution_feasibility(
        result.solution,
        config.base_energy_rate,
        config.payload_energy_rate,
        config.enforce_no_fly_zones,
    )

    return AlgorithmMetrics(
        instance_id=instance.instance_id,
        algorithm="Simulated Annealing",
        status=result.status,
        feasible=feasibility.feasible,
        objective_value=result.best_objective if feasibility.feasible else float("inf"),
        penalized_cost=result.best_penalized_cost,
        runtime_seconds=result.runtime_seconds,
        total_distance=solution_distance(result.solution),
        iterations=result.iterations_executed,
        feasibility_messages=" | ".join(feasibility.messages),
    )


def _convergence_from_branch_and_bound(
    instance_id: str,
    result: BranchAndBoundResult,
) -> list[ConvergencePoint]:
    """Convert exact-solver node logs to convergence points."""

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


def _convergence_from_ga(
    instance_id: str,
    result: GeneticAlgorithmResult,
) -> list[ConvergencePoint]:
    """Convert GA generation logs to convergence points."""

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


def _convergence_from_sa(
    instance_id: str,
    result: SimulatedAnnealingResult,
) -> list[ConvergencePoint]:
    """Convert SA annealing logs to convergence points."""

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
