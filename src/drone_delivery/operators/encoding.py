"""Solution encodings for exact methods and metaheuristics.

The Genetic Algorithm uses a route-based chromosome:

    ((1, 5, 2), (3,), (4, 6))

Each inner tuple is the ordered sequence of customer ids assigned to one drone.
The index of the inner tuple corresponds to the index of the drone in
``ProblemInstance.drones``. This encoding is natural for drone routing because
it stores both assignment decisions and visit-order decisions in one object.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from drone_delivery.core.entities import ProblemInstance
from drone_delivery.core.solution import Route, Solution


@dataclass(frozen=True)
class RouteChromosome:
    """Route-based chromosome for drone delivery metaheuristics.

    This encoding is adapted to drone delivery because a single object stores:

    - the assignment of customers to drones,
    - the order in which each drone visits its assigned customers,
    - empty routes for drones that are available but unused.

    This is more suitable than a plain permutation for drone routing because
    payload and battery feasibility are route-specific constraints.
    """

    routes: tuple[tuple[int, ...], ...]

    @property
    def customer_ids(self) -> list[int]:
        """Return all encoded customer ids in route order."""

        return [customer_id for route in self.routes for customer_id in route]

    def copy_with_routes(self, routes: tuple[tuple[int, ...], ...]) -> "RouteChromosome":
        """Return a new chromosome with replaced routes."""

        return RouteChromosome(routes=routes)

    def to_solution(self, instance: ProblemInstance) -> Solution:
        """Convert the chromosome into the shared ``Solution`` domain object."""

        solution_routes: list[Route] = []
        for drone, route_ids in zip(instance.drones, self.routes):
            customers = tuple(instance.customer_by_id(customer_id) for customer_id in route_ids)
            if customers:
                solution_routes.append(
                    Route(drone=drone, depot=instance.depot, customers=customers)
                )

        return Solution(instance=instance, routes=tuple(solution_routes))


def create_empty_chromosome(instance: ProblemInstance) -> RouteChromosome:
    """Create a chromosome with one empty route for each drone."""

    return RouteChromosome(routes=tuple(() for _ in instance.drones))


def create_random_chromosome(
    instance: ProblemInstance,
    rng: random.Random,
) -> RouteChromosome:
    """Create a random chromosome before repair.

    Customers are shuffled and assigned to random drones. The repair operator is
    responsible for improving feasibility afterward.
    """

    routes: list[list[int]] = [[] for _ in instance.drones]
    customer_ids = [customer.id for customer in instance.customers]
    rng.shuffle(customer_ids)

    for customer_id in customer_ids:
        drone_index = rng.randrange(len(routes))
        routes[drone_index].append(customer_id)

    return RouteChromosome(routes=tuple(tuple(route) for route in routes))


def solution_to_chromosome(solution: Solution, instance: ProblemInstance) -> RouteChromosome:
    """Encode a domain ``Solution`` as a route chromosome.

    The result always contains one route per drone in ``instance.drones``. This
    makes it safe to pass solutions from exact methods into metaheuristic
    operators for polishing or comparison.
    """

    routes_by_drone = {route.drone.id: tuple(route.customer_ids) for route in solution.routes}
    return RouteChromosome(
        routes=tuple(routes_by_drone.get(drone.id, ()) for drone in instance.drones)
    )


def flatten_chromosome(chromosome: RouteChromosome) -> list[int]:
    """Return a permutation-like view of a route chromosome."""

    return chromosome.customer_ids


def build_chromosome_from_sequence(
    sequence: list[int],
    route_lengths: list[int],
    number_of_routes: int,
) -> RouteChromosome:
    """Build a route chromosome from a flat sequence and route lengths."""

    routes: list[tuple[int, ...]] = []
    cursor = 0

    for route_index in range(number_of_routes):
        route_length = route_lengths[route_index] if route_index < len(route_lengths) else 0
        routes.append(tuple(sequence[cursor : cursor + route_length]))
        cursor += route_length

    if cursor < len(sequence) and routes:
        routes[-1] = routes[-1] + tuple(sequence[cursor:])

    return RouteChromosome(routes=tuple(routes))
