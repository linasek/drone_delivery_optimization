import random

from drone_delivery.algorithms.metaheuristics import (
    GeneticAlgorithmConfig,
    solve_genetic_algorithm,
)
from drone_delivery.core import Customer, Depot, Drone, ProblemInstance
from drone_delivery.operators import RouteChromosome, repair_chromosome


def build_ga_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id="ga_small",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=1.0),
            Customer(id=2, x=2.0, y=0.0, demand=1.0),
            Customer(id=3, x=0.0, y=2.0, demand=1.0),
        ),
        drones=(
            Drone(id=1, payload_capacity=5.0, battery_capacity=100.0),
            Drone(id=2, payload_capacity=5.0, battery_capacity=100.0),
        ),
    )


def test_repair_restores_customer_coverage():
    instance = build_ga_instance()
    invalid = RouteChromosome(routes=((1, 1, 99), ()))

    repaired = repair_chromosome(invalid, instance, random.Random(4))
    assigned = sorted(repaired.customer_ids)

    assert assigned == [1, 2, 3]


def test_genetic_algorithm_returns_feasible_solution_on_small_instance():
    instance = build_ga_instance()
    result = solve_genetic_algorithm(
        instance,
        GeneticAlgorithmConfig(
            population_size=20,
            generations=30,
            no_improvement_generations=10,
            random_seed=5,
            payload_energy_rate=0.0,
        ),
    )

    is_valid, messages = result.solution.validate_feasibility()

    assert result.found_feasible_solution
    assert is_valid, messages
    assert result.best_objective < float("inf")
    assert result.generations_executed > 0
    assert result.logs
