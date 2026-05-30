"""Crossover operators preserving or repairing customer coverage."""

from __future__ import annotations

import random

from drone_delivery.core.entities import ProblemInstance
from drone_delivery.operators.encoding import (
    RouteChromosome,
    build_chromosome_from_sequence,
)
from drone_delivery.operators.repair import repair_chromosome


def order_crossover(
    parent_a: RouteChromosome,
    parent_b: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Route-aware order crossover for drone delivery.

    The operator first applies classical order crossover to the flattened
    customer sequence. It then splits the child sequence according to route
    lengths inherited from one parent and finally calls the repair operator.

    This preserves useful ordering information while allowing the repair step
    to restore drone-specific feasibility.
    """

    sequence_a = parent_a.customer_ids
    sequence_b = parent_b.customer_ids
    customer_count = len(instance.customers)

    if customer_count <= 1:
        return repair_chromosome(
            parent_a,
            instance,
            rng,
            base_energy_rate,
            payload_energy_rate,
            penalty_weight,
        )

    child_sequence = _order_crossover_sequence(sequence_a, sequence_b, rng)

    # Route lengths encode a rough assignment pattern. Choosing the source
    # parent randomly gives the child either parent's drone-allocation shape.
    length_source = parent_a if rng.random() < 0.5 else parent_b
    route_lengths = [len(route) for route in length_source.routes]
    child = build_chromosome_from_sequence(
        child_sequence,
        route_lengths,
        len(instance.drones),
    )

    return repair_chromosome(
        child,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def route_exchange_crossover(
    parent_a: RouteChromosome,
    parent_b: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Exchange complete drone routes between parents.

    This crossover is adapted to drone delivery because a good parent route is
    often a coherent feasible mission: it already balances visit order,
    payload, and battery use for one drone. The operator copies some complete
    routes from one parent, fills the remaining customers using the order of
    the second parent, and repairs capacity violations.
    """

    number_of_routes = len(instance.drones)
    selected_routes = set(rng.sample(range(number_of_routes), rng.randrange(number_of_routes + 1)))
    child_routes: list[list[int]] = [[] for _ in instance.drones]
    used_customers: set[int] = set()

    for route_index in selected_routes:
        route = parent_a.routes[route_index] if route_index < len(parent_a.routes) else ()
        child_routes[route_index] = list(route)
        used_customers.update(route)

    fill_route_index = 0
    for customer_id in parent_b.customer_ids:
        if customer_id in used_customers:
            continue
        while fill_route_index in selected_routes and fill_route_index < number_of_routes - 1:
            fill_route_index += 1
        child_routes[fill_route_index].append(customer_id)
        used_customers.add(customer_id)

    return repair_chromosome(
        RouteChromosome(routes=tuple(tuple(route) for route in child_routes)),
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def _order_crossover_sequence(
    sequence_a: list[int],
    sequence_b: list[int],
    rng: random.Random,
) -> list[int]:
    """Apply order crossover to two customer permutations."""

    size = len(sequence_a)
    start, end = sorted(rng.sample(range(size), 2))
    child: list[int | None] = [None] * size
    child[start : end + 1] = sequence_a[start : end + 1]

    used = {customer_id for customer_id in child if customer_id is not None}
    fill_values = [customer_id for customer_id in sequence_b if customer_id not in used]
    fill_index = 0

    for index in range(size):
        if child[index] is None:
            child[index] = fill_values[fill_index]
            fill_index += 1

    return [customer_id for customer_id in child if customer_id is not None]

