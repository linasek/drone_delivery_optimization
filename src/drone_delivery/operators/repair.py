"""Repair operators for drone-routing chromosomes.

Genetic operators can create invalid chromosomes: duplicated customers, missing
customers, routes assigned to too few or too many drones, or routes that exceed
payload/battery capacity. Repair operators attempt to restore a useful
candidate solution before fitness evaluation.
"""

from __future__ import annotations

import math
import random

from drone_delivery.core.entities import ProblemInstance
from drone_delivery.core.solution import Route
from drone_delivery.operators.encoding import RouteChromosome


def repair_chromosome(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Repair customer coverage and improve route feasibility.

    The repair procedure has three academic purposes:

    1. ensure every customer appears exactly once,
    2. keep one route position for each available drone,
    3. insert or relocate customers using the lowest penalized energy increase.

    If an instance is genuinely infeasible, repair cannot create a feasible
    solution. In that case, it returns the best corrected chromosome it can, and
    the fitness function assigns a penalty.
    """

    normalized_routes, missing = _remove_duplicates_and_unknowns(chromosome, instance)

    # Insert missing customers in descending demand order. High-demand packages
    # are harder to place, so assigning them first usually creates better
    # chromosomes.
    missing.sort(
        key=lambda customer_id: instance.customer_by_id(customer_id).demand,
        reverse=True,
    )
    for customer_id in missing:
        normalized_routes = _insert_customer_best_position(
            routes=normalized_routes,
            customer_id=customer_id,
            instance=instance,
            base_energy_rate=base_energy_rate,
            payload_energy_rate=payload_energy_rate,
            penalty_weight=penalty_weight,
        )

    # A few relocation passes often fix capacity violations caused by crossover
    # or random initialization. The loop is deliberately bounded so repair
    # remains cheap compared with the GA search itself.
    for _ in range(3):
        improved = _relocate_from_infeasible_routes(
            normalized_routes,
            instance,
            rng,
            base_energy_rate,
            payload_energy_rate,
            penalty_weight,
        )
        if improved == normalized_routes:
            break
        normalized_routes = improved

    return RouteChromosome(routes=tuple(tuple(route) for route in normalized_routes))


def repair_infeasible_chromosome(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> RouteChromosome:
    """Academic alias emphasizing the purpose of repair.

    This function is intentionally explicit for report writing: genetic and
    neighborhood operators may violate customer coverage, payload capacity, or
    battery capacity. The repair operator restores customer coverage first,
    then relocates customers away from overloaded routes when possible.
    """

    return repair_chromosome(
        chromosome,
        instance,
        rng,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )


def is_chromosome_feasible(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
) -> bool:
    """Return True if customer coverage, payload, and battery constraints hold."""

    solution = chromosome.to_solution(instance)
    is_valid, _ = solution.validate_feasibility(base_energy_rate, payload_energy_rate)
    return is_valid


def chromosome_penalty(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    penalty_weight: float = 10_000.0,
) -> float:
    """Return a scalar penalty for remaining infeasibilities."""

    penalty = 0.0
    required = instance.customer_ids
    assigned = chromosome.customer_ids
    assigned_set = set(assigned)

    missing_count = len(required - assigned_set)
    unknown_count = len(assigned_set - required)
    duplicate_count = len(assigned) - len(assigned_set)
    penalty += penalty_weight * (missing_count + unknown_count + duplicate_count)

    for drone, route_ids in zip(instance.drones, chromosome.routes):
        customers = tuple(instance.customer_by_id(customer_id) for customer_id in route_ids)
        route = Route(drone=drone, depot=instance.depot, customers=customers)
        payload_excess = max(0.0, route.total_payload() - drone.payload_capacity)
        energy_excess = max(
            0.0,
            route.total_energy(base_energy_rate, payload_energy_rate)
            - drone.battery_capacity,
        )
        penalty += penalty_weight * payload_excess
        penalty += penalty_weight * energy_excess

    return penalty


def _remove_duplicates_and_unknowns(
    chromosome: RouteChromosome,
    instance: ProblemInstance,
) -> tuple[list[list[int]], list[int]]:
    """Keep the first valid occurrence of each customer and report missing ids."""

    required = instance.customer_ids
    seen: set[int] = set()
    routes: list[list[int]] = []

    for route_index in range(len(instance.drones)):
        source_route = chromosome.routes[route_index] if route_index < len(chromosome.routes) else ()
        cleaned_route: list[int] = []
        for customer_id in source_route:
            if customer_id in required and customer_id not in seen:
                cleaned_route.append(customer_id)
                seen.add(customer_id)
        routes.append(cleaned_route)

    missing = sorted(required - seen)
    return routes, missing


def _insert_customer_best_position(
    routes: list[list[int]],
    customer_id: int,
    instance: ProblemInstance,
    base_energy_rate: float,
    payload_energy_rate: float,
    penalty_weight: float,
) -> list[list[int]]:
    """Insert one missing customer where penalized route cost is minimized."""

    best_score = math.inf
    best_routes: list[list[int]] | None = None

    for route_index, route in enumerate(routes):
        for position in range(len(route) + 1):
            candidate = [list(existing) for existing in routes]
            candidate[route_index].insert(position, customer_id)
            score = _total_penalized_route_cost(
                candidate,
                instance,
                base_energy_rate,
                payload_energy_rate,
                penalty_weight,
            )
            if score < best_score:
                best_score = score
                best_routes = candidate

    if best_routes is None:
        return routes

    return best_routes


def _relocate_from_infeasible_routes(
    routes: list[list[int]],
    instance: ProblemInstance,
    rng: random.Random,
    base_energy_rate: float,
    payload_energy_rate: float,
    penalty_weight: float,
) -> list[list[int]]:
    """Try to move one customer out of an infeasible route."""

    current_score = _total_penalized_route_cost(
        routes,
        instance,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )
    best_score = current_score
    best_routes = [list(route) for route in routes]

    route_indices = list(range(len(routes)))
    rng.shuffle(route_indices)

    for source_index in route_indices:
        if not _route_has_violation(
            routes[source_index],
            source_index,
            instance,
            base_energy_rate,
            payload_energy_rate,
        ):
            continue

        for customer_position, customer_id in enumerate(routes[source_index]):
            for target_index in range(len(routes)):
                if target_index == source_index:
                    continue
                for insert_position in range(len(routes[target_index]) + 1):
                    candidate = [list(route) for route in routes]
                    candidate[source_index].pop(customer_position)
                    candidate[target_index].insert(insert_position, customer_id)
                    score = _total_penalized_route_cost(
                        candidate,
                        instance,
                        base_energy_rate,
                        payload_energy_rate,
                        penalty_weight,
                    )
                    if score < best_score:
                        best_score = score
                        best_routes = candidate

    return best_routes


def _route_has_violation(
    route_ids: list[int],
    route_index: int,
    instance: ProblemInstance,
    base_energy_rate: float,
    payload_energy_rate: float,
) -> bool:
    """Return True if one route exceeds payload or battery capacity."""

    drone = instance.drones[route_index]
    customers = tuple(instance.customer_by_id(customer_id) for customer_id in route_ids)
    route = Route(drone=drone, depot=instance.depot, customers=customers)
    return not route.is_feasible(base_energy_rate, payload_energy_rate)


def _total_penalized_route_cost(
    routes: list[list[int]],
    instance: ProblemInstance,
    base_energy_rate: float,
    payload_energy_rate: float,
    penalty_weight: float,
) -> float:
    """Compute total energy plus payload/battery violation penalties."""

    chromosome = RouteChromosome(routes=tuple(tuple(route) for route in routes))
    energy = chromosome.to_solution(instance).total_energy(
        base_energy_rate,
        payload_energy_rate,
    )
    return energy + chromosome_penalty(
        chromosome,
        instance,
        base_energy_rate,
        payload_energy_rate,
        penalty_weight,
    )
