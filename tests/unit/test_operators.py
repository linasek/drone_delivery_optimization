import random

from drone_delivery.core import Customer, Depot, Drone, ProblemInstance
from drone_delivery.operators import (
    RouteChromosome,
    flatten_chromosome,
    insertion_mutation,
    insertion_operator,
    is_chromosome_feasible,
    merge_mutation,
    mutate_chromosome,
    order_crossover,
    repair_infeasible_chromosome,
    route_exchange_crossover,
    route_merge_operator,
    route_split_operator,
    split_mutation,
    swap_mutation,
    swap_operator,
)


def build_operator_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id="operators",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=1.0),
            Customer(id=2, x=2.0, y=0.0, demand=1.0),
            Customer(id=3, x=0.0, y=1.0, demand=1.0),
            Customer(id=4, x=0.0, y=2.0, demand=1.0),
        ),
        drones=(
            Drone(id=1, payload_capacity=4.0, battery_capacity=100.0),
            Drone(id=2, payload_capacity=4.0, battery_capacity=100.0),
            Drone(id=3, payload_capacity=4.0, battery_capacity=100.0),
        ),
    )


def assert_valid_operator_output(chromosome: RouteChromosome, instance: ProblemInstance) -> None:
    assert sorted(flatten_chromosome(chromosome)) == [1, 2, 3, 4]
    assert is_chromosome_feasible(chromosome, instance)


def test_repair_handles_duplicate_missing_and_unknown_customers():
    instance = build_operator_instance()
    chromosome = RouteChromosome(routes=((1, 1, 99), (3,), ()))

    repaired = repair_infeasible_chromosome(chromosome, instance, random.Random(1))

    assert_valid_operator_output(repaired, instance)


def test_swap_insertion_split_and_merge_operators_preserve_feasibility():
    instance = build_operator_instance()
    rng = random.Random(2)
    chromosome = RouteChromosome(routes=((1, 2, 3, 4), (), ()))

    split_child = route_split_operator(chromosome, instance, rng)
    swap_child = swap_operator(split_child, instance, rng)
    insertion_child = insertion_operator(swap_child, instance, rng)
    merge_child = route_merge_operator(insertion_child, instance, rng)

    assert_valid_operator_output(split_child, instance)
    assert_valid_operator_output(swap_child, instance)
    assert_valid_operator_output(insertion_child, instance)
    assert_valid_operator_output(merge_child, instance)


def test_mutation_wrappers_preserve_feasibility():
    instance = build_operator_instance()
    rng = random.Random(3)
    chromosome = RouteChromosome(routes=((1, 2), (3, 4), ()))

    for operator in (
        swap_mutation,
        insertion_mutation,
        split_mutation,
        merge_mutation,
    ):
        child = operator(chromosome, instance, rng)
        assert_valid_operator_output(child, instance)

    mutated = mutate_chromosome(chromosome, instance, rng, mutation_rate=1.0)
    assert_valid_operator_output(mutated, instance)


def test_crossover_operators_preserve_customer_coverage_and_feasibility():
    instance = build_operator_instance()
    rng = random.Random(4)
    parent_a = RouteChromosome(routes=((1, 2), (3,), (4,)))
    parent_b = RouteChromosome(routes=((4, 3), (2,), (1,)))

    order_child = order_crossover(parent_a, parent_b, instance, rng)
    route_child = route_exchange_crossover(parent_a, parent_b, instance, rng)

    assert_valid_operator_output(order_child, instance)
    assert_valid_operator_output(route_child, instance)
