from pathlib import Path

from drone_delivery.algorithms.exact import BranchAndBoundConfig
from drone_delivery.algorithms.metaheuristics import (
    GeneticAlgorithmConfig,
    SimulatedAnnealingConfig,
)
from drone_delivery.core import Customer, Depot, Drone, ProblemInstance
from drone_delivery.evaluation import EvaluationConfig, compare_algorithms


def build_evaluation_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id="evaluation_small",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=1.0),
            Customer(id=2, x=2.0, y=0.0, demand=1.0),
            Customer(id=3, x=0.0, y=1.0, demand=1.0),
        ),
        drones=(
            Drone(id=1, payload_capacity=5.0, battery_capacity=100.0),
            Drone(id=2, payload_capacity=5.0, battery_capacity=100.0),
        ),
    )


def test_compare_algorithms_exports_metrics_and_convergence_csv():
    instance = build_evaluation_instance()
    report = compare_algorithms(
        instance,
        EvaluationConfig(
            branch_and_bound_config=BranchAndBoundConfig(
                base_energy_rate=1.0,
                payload_energy_rate=0.0,
                max_nodes=1000,
            ),
            genetic_algorithm_config=GeneticAlgorithmConfig(
                population_size=12,
                generations=10,
                no_improvement_generations=5,
                random_seed=2,
                payload_energy_rate=0.0,
            ),
            simulated_annealing_config=SimulatedAnnealingConfig(
                initial_temperature=20.0,
                cooling_rate=0.90,
                iterations_per_temperature=5,
                max_iterations=40,
                no_improvement_iterations=20,
                random_seed=3,
                payload_energy_rate=0.0,
            ),
            payload_energy_rate=0.0,
        ),
    )

    algorithms = {metric.algorithm for metric in report.metrics}
    assert algorithms == {"Branch and Bound", "Genetic Algorithm", "Simulated Annealing"}
    assert all(metric.runtime_seconds >= 0.0 for metric in report.metrics)
    assert all(metric.feasible for metric in report.metrics)
    assert report.convergence

    output_dir = Path("outputs") / "tables"
    metrics_path, convergence_path = report.export(
        output_dir=output_dir,
        metrics_file_name="test_metrics.csv",
        convergence_file_name="test_convergence.csv",
    )

    assert metrics_path.exists()
    assert convergence_path.exists()
    assert metrics_path.stat().st_size > 0
    assert convergence_path.stat().st_size > 0
