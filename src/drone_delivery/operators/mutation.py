"""Mutation operators adapted to route and drone assignment changes."""

from __future__ import annotations

import random

from drone_delivery.core.entities import ProblemInstance
from drone_delivery.operators.encoding import RouteChromosome
from drone_delivery.operators.neighborhood import (
    insertion_operator,
    route_merge_operator,
    route_split_operator,
    swap_operator,
)
from drone_delivery.operators.repair import repair_chromosome


def mutate_chromosome(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    mutation_rate: float,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Mutate a route chromosome using drone-routing moves.

    The operator randomly selects one of six local changes:

    - swap two customers,
    - relocate one customer,
    - reverse a subsequence inside one route,
    - move one customer between drones,
    - split a long route,
    - merge two short routes.

    Every mutation is followed by repair to preserve customer coverage.
    """

    if rng.random() > mutation_rate:
        return chromosome

    routes = [list(route) for route in chromosome.routes]
    if not any(routes):
        return chromosome

    mutation = rng.choice(
        [
            _swap_customers,
            _relocate_customer,
            _reverse_subroute,
            _move_between_routes,
            _split_route,
            _merge_routes,
        ]
    )
    mutation(routes, rng)

    return repair_chromosome(
        RouteChromosome(routes=tuple(tuple(route) for route in routes)),
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def _non_empty_route_indices(routes: list[list[int]]) -> list[int]:
    """Return route indices that currently contain at least one customer."""

    return [index for index, route in enumerate(routes) if route]


def _swap_customers(routes: list[list[int]], rng: random.Random) -> None:
    """Swap two customers, possibly from different drone routes."""

    positions = [
        (route_index, customer_index)
        for route_index, route in enumerate(routes)
        for customer_index in range(len(route))
    ]
    if len(positions) < 2:
        return

    first, second = rng.sample(positions, 2)
    r1, c1 = first
    r2, c2 = second
    routes[r1][c1], routes[r2][c2] = routes[r2][c2], routes[r1][c1]


def _relocate_customer(routes: list[list[int]], rng: random.Random) -> None:
    """Move one customer to a different position, possibly in the same route."""

    source_indices = _non_empty_route_indices(routes)
    if not source_indices:
        return

    source = rng.choice(source_indices)
    customer_position = rng.randrange(len(routes[source]))
    customer_id = routes[source].pop(customer_position)

    target = rng.randrange(len(routes))
    insert_position = rng.randrange(len(routes[target]) + 1)
    routes[target].insert(insert_position, customer_id)


def swap_mutation(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Mutation wrapper around the drone-specific swap operator."""

    return swap_operator(
        chromosome,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def insertion_mutation(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Mutation wrapper around best-position customer insertion."""

    return insertion_operator(
        chromosome,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def split_mutation(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Mutation wrapper that splits one long route if an unused drone exists."""

    return route_split_operator(
        chromosome,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def merge_mutation(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Mutation wrapper that merges two routes and repairs violations."""

    return route_merge_operator(
        chromosome,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def _reverse_subroute(routes: list[list[int]], rng: random.Random) -> None:
    """Reverse a consecutive sequence inside one drone route."""

    candidates = [index for index, route in enumerate(routes) if len(route) >= 2]
    if not candidates:
        return

    route_index = rng.choice(candidates)
    route = routes[route_index]
    start, end = sorted(rng.sample(range(len(route)), 2))
    route[start : end + 1] = reversed(route[start : end + 1])


def _move_between_routes(routes: list[list[int]], rng: random.Random) -> None:
    """Move one customer from one drone route to another drone route."""

    source_indices = _non_empty_route_indices(routes)
    if not source_indices or len(routes) < 2:
        return

    source = rng.choice(source_indices)
    target_choices = [index for index in range(len(routes)) if index != source]
    target = rng.choice(target_choices)

    customer_position = rng.randrange(len(routes[source]))
    customer_id = routes[source].pop(customer_position)
    insert_position = rng.randrange(len(routes[target]) + 1)
    routes[target].insert(insert_position, customer_id)


def _split_route(routes: list[list[int]], rng: random.Random) -> None:
    """Split one route into an empty route when possible."""

    splittable = [index for index, route in enumerate(routes) if len(route) >= 2]
    empty = [index for index, route in enumerate(routes) if not route]
    if not splittable or not empty:
        return

    source = rng.choice(splittable)
    target = rng.choice(empty)
    cut = rng.randrange(1, len(routes[source]))
    routes[target] = routes[source][cut:]
    routes[source] = routes[source][:cut]


def _merge_routes(routes: list[list[int]], rng: random.Random) -> None:
    """Merge two non-empty routes into one route."""

    non_empty = _non_empty_route_indices(routes)
    if len(non_empty) < 2:
        return

    first, second = rng.sample(non_empty, 2)
    routes[first].extend(routes[second])
    routes[second] = []
