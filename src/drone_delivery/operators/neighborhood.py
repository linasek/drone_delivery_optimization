"""Neighborhood moves for local search and simulated annealing.

The neighborhood operators modify the route-based chromosome used throughout
the metaheuristics. Each move changes either the assignment of a customer to a
drone or the visit order inside a drone route.
"""

from __future__ import annotations

import random

from drone_delivery.core.entities import ProblemInstance
from drone_delivery.operators.encoding import RouteChromosome
from drone_delivery.operators.repair import repair_chromosome


def generate_neighbor(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Generate and repair one neighboring chromosome.

    Simulated Annealing needs a local move that changes the current solution
    slightly. The selected move is random, but every move is specialized for
    drone routing:

    - swap two customers,
    - relocate one customer to another position,
    - reverse a route segment,
    - move a customer from one drone route to another,
    - exchange tails between two drone routes.
    """

    routes = [list(route) for route in chromosome.routes]
    if not any(routes):
        return repair_chromosome(
            chromosome,
            instance,
            rng,
            base_energy_rate,
            payload_energy_rate,
            penalty_weight,
        )

    move = rng.choice(
        [
            _swap_two_customers_in_place,
            _relocate_customer_in_place,
            _reverse_route_segment_in_place,
            _move_customer_between_drones_in_place,
            _exchange_route_tails_in_place,
        ]
    )
    move(routes, rng)

    return repair_chromosome(
        RouteChromosome(routes=tuple(tuple(route) for route in routes)),
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def swap_operator(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Swap two customers and repair the resulting drone routes.

    Swapping is useful for drone delivery because changing the order can reduce
    payload-dependent energy: serving heavier packages earlier or later changes
    how much weight is carried on each arc.
    """

    routes = [list(route) for route in chromosome.routes]
    _swap_two_customers_in_place(routes, rng)
    return _repair_routes(
        routes,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def insertion_operator(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Remove one customer and insert it in the best feasible position found.

    Insertion is adapted to drone routing because it can move a package from an
    overloaded route to a route with remaining payload or battery capacity.
    """

    routes = [list(route) for route in chromosome.routes]
    positions = _all_customer_positions(routes)
    if not positions:
        return chromosome

    source_route, source_position = rng.choice(positions)
    customer_id = routes[source_route].pop(source_position)
    best_routes = _best_insertion_for_customer(
        routes,
        customer_id,
        instance,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )
    return _repair_routes(
        best_routes,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def route_split_operator(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Split one long drone route into two routes.

    This operator is drone-specific because long routes are often infeasible
    due to battery or payload limits. Splitting a route can convert one
    overloaded mission into two feasible missions flown by separate drones.
    """

    routes = [list(route) for route in chromosome.routes]
    splittable = [index for index, route in enumerate(routes) if len(route) >= 2]
    targets = [index for index, route in enumerate(routes) if not route]
    if not splittable or not targets:
        return chromosome

    source = rng.choice(splittable)
    target = rng.choice(targets)
    cut = rng.randrange(1, len(routes[source]))
    routes[target] = routes[source][cut:]
    routes[source] = routes[source][:cut]

    return _repair_routes(
        routes,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def route_merge_operator(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Merge two drone routes and repair if capacity is violated.

    Merging is useful when two short routes can be served by one drone, reducing
    depot returns and potentially saving energy. The repair step protects
    against payload and battery violations after the merge.
    """

    routes = [list(route) for route in chromosome.routes]
    non_empty = _non_empty_route_indices(routes)
    if len(non_empty) < 2:
        return chromosome

    first, second = rng.sample(non_empty, 2)
    if rng.random() < 0.5:
        routes[first] = routes[first] + routes[second]
    else:
        routes[first] = routes[second] + routes[first]
    routes[second] = []

    return _repair_routes(
        routes,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def _swap_two_customers_in_place(routes: list[list[int]], rng: random.Random) -> None:
    """Swap two customer visits, possibly from different drones."""

    positions = _all_customer_positions(routes)
    if len(positions) < 2:
        return

    first, second = rng.sample(positions, 2)
    r1, c1 = first
    r2, c2 = second
    routes[r1][c1], routes[r2][c2] = routes[r2][c2], routes[r1][c1]


def _relocate_customer_in_place(routes: list[list[int]], rng: random.Random) -> None:
    """Remove one customer and insert it into a new route position."""

    source_routes = _non_empty_route_indices(routes)
    if not source_routes:
        return

    source = rng.choice(source_routes)
    source_position = rng.randrange(len(routes[source]))
    customer_id = routes[source].pop(source_position)

    target = rng.randrange(len(routes))
    target_position = rng.randrange(len(routes[target]) + 1)
    routes[target].insert(target_position, customer_id)


def _reverse_route_segment_in_place(routes: list[list[int]], rng: random.Random) -> None:
    """Reverse a consecutive customer segment inside one route."""

    candidates = [index for index, route in enumerate(routes) if len(route) >= 2]
    if not candidates:
        return

    route_index = rng.choice(candidates)
    route = routes[route_index]
    start, end = sorted(rng.sample(range(len(route)), 2))
    route[start : end + 1] = reversed(route[start : end + 1])


def _move_customer_between_drones_in_place(routes: list[list[int]], rng: random.Random) -> None:
    """Move a customer from one drone route to a different drone route."""

    source_routes = _non_empty_route_indices(routes)
    if len(routes) < 2 or not source_routes:
        return

    source = rng.choice(source_routes)
    target = rng.choice([index for index in range(len(routes)) if index != source])
    customer_position = rng.randrange(len(routes[source]))
    customer_id = routes[source].pop(customer_position)
    insert_position = rng.randrange(len(routes[target]) + 1)
    routes[target].insert(insert_position, customer_id)


def _exchange_route_tails_in_place(routes: list[list[int]], rng: random.Random) -> None:
    """Exchange route suffixes between two drones.

    This move changes customer assignment and visit order at the same time,
    which helps the search escape local optima that simple swaps cannot leave.
    """

    if len(routes) < 2:
        return

    first, second = rng.sample(range(len(routes)), 2)
    cut_first = rng.randrange(len(routes[first]) + 1)
    cut_second = rng.randrange(len(routes[second]) + 1)

    first_head = routes[first][:cut_first]
    first_tail = routes[first][cut_first:]
    second_head = routes[second][:cut_second]
    second_tail = routes[second][cut_second:]

    routes[first] = first_head + second_tail
    routes[second] = second_head + first_tail


def _all_customer_positions(routes: list[list[int]]) -> list[tuple[int, int]]:
    """Return all route/customer index pairs."""

    return [
        (route_index, customer_index)
        for route_index, route in enumerate(routes)
        for customer_index in range(len(route))
    ]


def _non_empty_route_indices(routes: list[list[int]]) -> list[int]:
    """Return indices of routes containing at least one customer."""

    return [index for index, route in enumerate(routes) if route]


def _best_insertion_for_customer(
    routes: list[list[int]],
    customer_id: int,
    instance: ProblemInstance,
    base_energy_rate: float,
    payload_energy_rate: float,
    penalty_weight: float,
) -> list[list[int]]:
    """Find the insertion position with the lowest penalized solution cost."""

    best_score = float("inf")
    best_routes = [list(route) for route in routes]

    for route_index, route in enumerate(routes):
        for position in range(len(route) + 1):
            candidate = [list(existing) for existing in routes]
            candidate[route_index].insert(position, customer_id)
            repaired = RouteChromosome(routes=tuple(tuple(item) for item in candidate))
            solution = repaired.to_solution(instance)
            cost = solution.total_energy(base_energy_rate, payload_energy_rate)
            cost += _route_constraint_penalty(
                repaired,
                instance,
                base_energy_rate,
                payload_energy_rate,
                penalty_weight,
            )
            if cost < best_score:
                best_score = cost
                best_routes = candidate

    return best_routes


def _route_constraint_penalty(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    base_energy_rate: float,
    payload_energy_rate: float,
    penalty_weight: float,
) -> float:
    """Local import avoids a module cycle with the repair module."""

    from drone_delivery.operators.repair import chromosome_penalty

    return chromosome_penalty(
        chromosome,
        instance,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def _repair_routes(
    routes: list[list[int]],
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float,
    payload_energy_rate: float,
    penalty_weight: float,
) -> RouteChromosome:
    """Convert mutable routes to a chromosome and repair infeasibilities."""

    return repair_chromosome(
        RouteChromosome(routes=tuple(tuple(route) for route in routes)),
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )
