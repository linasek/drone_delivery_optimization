import random

from drone_delivery.algorithms.metaheuristics import (
    SimulatedAnnealingConfig,
    solve_simulated_annealing,
)
from drone_delivery.core import Customer, Depot, Drone, ProblemInstance
from drone_delivery.operators import RouteChromosome, generate_neighbor


def build_sa_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id="sa_small",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=1.0),
            Customer(id=2, x=2.0, y=0.0, demand=1.0),
            Customer(id=3, x=0.0, y=2.0, demand=1.0),
            Customer(id=4, x=0.0, y=3.0, demand=1.0),
        ),
        drones=(
            Drone(id=1, payload_capacity=5.0, battery_capacity=100.0),
            Drone(id=2, payload_capacity=5.0, battery_capacity=100.0),
        ),
    )


def test_neighborhood_generation_preserves_customer_coverage_after_repair():
    instance = build_sa_instance()
    chromosome = RouteChromosome(routes=((1, 2), (3, 4)))

    neighbor = generate_neighbor(chromosome, instance, random.Random(8))

    assert sorted(neighbor.customer_ids) == [1, 2, 3, 4]


def test_simulated_annealing_returns_feasible_solution_on_small_instance():
    instance = build_sa_instance()
    result = solve_simulated_annealing(
        instance,
        SimulatedAnnealingConfig(
            initial_temperature=50.0,
            cooling_rate=0.90,
            iterations_per_temperature=10,
            max_iterations=120,
            no_improvement_iterations=60,
            random_seed=9,
            payload_energy_rate=0.0,
        ),
    )

    is_valid, messages = result.solution.validate_feasibility()

    assert result.found_feasible_solution
    assert is_valid, messages
    assert result.best_objective < float("inf")
    assert result.iterations_executed > 0
    assert result.logs
