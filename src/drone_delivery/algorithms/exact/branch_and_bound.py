"""Branch and Bound exact solver for small drone delivery instances.

The solver performs an exact combinatorial search over:

1. which drone serves each customer, and
2. the visit order of customers inside each drone route.

At each branching step, one unassigned customer is inserted into every possible
position of every route. This enumerates all route partitions and permutations
without relying on an external MILP solver. The method is exact when no
``max_nodes`` or ``time_limit_seconds`` cutoff is used.

Branch and Bound is mainly intended for small benchmark instances. It is useful
academically because it provides optimal reference values against which the
Genetic Algorithm and Simulated Annealing methods can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable

from drone_delivery.core.entities import Customer, Depot, Location, NoFlyZone, ProblemInstance
from drone_delivery.core.solution import Route, Solution


@dataclass(frozen=True)
class BranchAndBoundConfig:
    """Configuration of the exact search.

    ``max_nodes`` and ``time_limit_seconds`` are optional safety limits. If they
    are omitted, the algorithm keeps searching until it proves optimality or
    proves infeasibility.
    """

    base_energy_rate: float = 1.0
    payload_energy_rate: float = 0.03
    max_nodes: int | None = None
    time_limit_seconds: float | None = None
    log_frequency: int = 1000


@dataclass(frozen=True)
class SearchNode:
    """Partial solution explored by Branch and Bound."""

    routes: tuple[tuple[Customer, ...], ...]
    next_customer_index: int
    current_objective: float
    lower_bound: float
    depth: int


@dataclass(frozen=True)
class NodeLog:
    """Compact information recorded during search for academic reporting."""

    explored_nodes: int
    depth: int
    current_objective: float
    lower_bound: float
    best_objective: float
    event: str


@dataclass
class BranchAndBoundResult:
    """Result returned by the exact solver."""

    solution: Solution | None
    best_objective: float
    runtime_seconds: float
    explored_nodes: int
    pruned_by_bound: int
    pruned_by_infeasibility: int
    completed_solutions: int
    status: str
    logs: list[NodeLog] = field(default_factory=list)

    @property
    def found_solution(self) -> bool:
        """Return True if the search found at least one feasible solution."""

        return self.solution is not None


class BranchAndBoundSolver:
    """Exact Branch and Bound solver for the drone delivery problem.

    The search tree is built incrementally. A node stores a partial set of
    drone routes and the index of the next customer to insert. Children are
    created by inserting that customer into every feasible position of every
    drone route.

    Pruning rules:
    - payload feasibility of partial routes,
    - battery feasibility of partial routes,
    - aggregate remaining payload capacity,
    - lower-bound dominance against the incumbent solution,
    - optional runtime or explored-node limits.
    """

    def __init__(
        self,
        instance: ProblemInstance,
        config: BranchAndBoundConfig | None = None,
    ) -> None:
        self.instance = instance
        self.config = config or BranchAndBoundConfig()

        # Branch on difficult customers first. High-demand and far-away
        # customers are more likely to violate payload/battery constraints, so
        # they help the search detect infeasibility early.
        self.ordered_customers = tuple(
            sorted(
                self.instance.customers,
                key=lambda customer: (
                    -customer.demand,
                    -self.instance.depot.distance_to(customer),
                    customer.id,
                ),
            )
        )

        self.best_solution: Solution | None = None
        self.best_objective = math.inf
        self.explored_nodes = 0
        self.pruned_by_bound = 0
        self.pruned_by_infeasibility = 0
        self.completed_solutions = 0
        self.logs: list[NodeLog] = []
        self._start_time = 0.0
        self._stopped_by_limit = False

    def solve(self) -> BranchAndBoundResult:
        """Run the Branch and Bound algorithm."""

        self._start_time = time.perf_counter()
        self._initialize_incumbent_with_greedy_solution()

        empty_routes = tuple(() for _ in self.instance.drones)
        root_objective = self._routes_objective(empty_routes)
        root_lower_bound = self._lower_bound(empty_routes, 0)
        root = SearchNode(
            routes=empty_routes,
            next_customer_index=0,
            current_objective=root_objective,
            lower_bound=root_lower_bound,
            depth=0,
        )

        self._search(root)

        runtime = time.perf_counter() - self._start_time
        if self._stopped_by_limit:
            status = "time_or_node_limit_reached"
        elif self.best_solution is None:
            status = "infeasible"
        else:
            status = "optimal"

        return BranchAndBoundResult(
            solution=self.best_solution,
            best_objective=self.best_objective,
            runtime_seconds=runtime,
            explored_nodes=self.explored_nodes,
            pruned_by_bound=self.pruned_by_bound,
            pruned_by_infeasibility=self.pruned_by_infeasibility,
            completed_solutions=self.completed_solutions,
            status=status,
            logs=self.logs,
        )

    def _search(self, node: SearchNode) -> None:
        """Recursively explore the search tree from one node."""

        if self._should_stop():
            self._stopped_by_limit = True
            return

        self.explored_nodes += 1
        self._log_node(node, event="explored")

        # Bounding rule: if even the optimistic lower bound is not better than
        # the best known feasible solution, this subtree cannot improve the
        # incumbent.
        if node.lower_bound >= self.best_objective:
            self.pruned_by_bound += 1
            self._log_node(node, event="pruned_by_bound")
            return

        if not self._remaining_capacity_can_cover_unassigned(node):
            self.pruned_by_infeasibility += 1
            self._log_node(node, event="pruned_by_remaining_capacity")
            return

        if node.next_customer_index == len(self.ordered_customers):
            self._evaluate_complete_node(node)
            return

        customer = self.ordered_customers[node.next_customer_index]
        children = list(self._branch_by_inserting_customer(node, customer))

        if not children:
            self.pruned_by_infeasibility += 1
            self._log_node(node, event="pruned_no_feasible_child")
            return

        # Best-first ordering inside the depth-first recursion. This does not
        # change exactness; it only helps find a strong incumbent early.
        children.sort(key=lambda child: child.lower_bound)
        for child in children:
            self._search(child)
            if self._stopped_by_limit:
                return

    def _branch_by_inserting_customer(
        self,
        node: SearchNode,
        customer: Customer,
    ) -> Iterable[SearchNode]:
        """Create children by inserting a customer into every route position."""

        for drone_index, route_customers in enumerate(node.routes):
            for position in range(len(route_customers) + 1):
                candidate_route = (
                    route_customers[:position]
                    + (customer,)
                    + route_customers[position:]
                )
                candidate_routes = (
                    node.routes[:drone_index]
                    + (candidate_route,)
                    + node.routes[drone_index + 1 :]
                )

                if not self._partial_routes_are_feasible(candidate_routes):
                    continue

                objective = self._routes_objective(candidate_routes)
                lower_bound = self._lower_bound(
                    candidate_routes,
                    node.next_customer_index + 1,
                )

                if lower_bound >= self.best_objective:
                    self.pruned_by_bound += 1
                    continue

                yield SearchNode(
                    routes=candidate_routes,
                    next_customer_index=node.next_customer_index + 1,
                    current_objective=objective,
                    lower_bound=lower_bound,
                    depth=node.depth + 1,
                )

    def _evaluate_complete_node(self, node: SearchNode) -> None:
        """Check a complete assignment and update the incumbent if improved."""

        solution = self._build_solution(node.routes, include_empty_routes=False)
        is_valid, _ = solution.validate_feasibility(
            base_rate=self.config.base_energy_rate,
            payload_rate=self.config.payload_energy_rate,
        )

        # No-fly-zone feasibility is checked only at complete routes. A partial
        # route arc may later be split by an inserted customer, so pruning a
        # partial node because of a no-fly crossing could incorrectly remove a
        # feasible final route.
        if is_valid:
            is_valid = self._solution_avoids_no_fly_zones(solution)

        if not is_valid:
            self.pruned_by_infeasibility += 1
            self._log_node(node, event="complete_infeasible")
            return

        self.completed_solutions += 1
        objective = solution.total_energy(
            base_rate=self.config.base_energy_rate,
            payload_rate=self.config.payload_energy_rate,
        )

        if objective < self.best_objective:
            self.best_solution = solution
            self.best_objective = objective
            self._log_node(node, event="new_incumbent")

    def _partial_routes_are_feasible(
        self,
        routes: tuple[tuple[Customer, ...], ...],
    ) -> bool:
        """Check pruning constraints that are safe for partial routes."""

        for drone, customers in zip(self.instance.drones, routes):
            route = Route(drone=drone, depot=self.instance.depot, customers=customers)

            if not route.is_payload_feasible():
                return False

            if not route.is_battery_feasible(
                base_rate=self.config.base_energy_rate,
                payload_rate=self.config.payload_energy_rate,
            ):
                return False

        return True

    def _remaining_capacity_can_cover_unassigned(self, node: SearchNode) -> bool:
        """Prune nodes that cannot fit the remaining demand into drone payloads."""

        unassigned = self.ordered_customers[node.next_customer_index :]
        if not unassigned:
            return True

        remaining_capacity_by_drone: list[float] = []
        for drone, route_customers in zip(self.instance.drones, node.routes):
            used_payload = sum(customer.demand for customer in route_customers)
            remaining_capacity_by_drone.append(drone.payload_capacity - used_payload)

        total_remaining_demand = sum(customer.demand for customer in unassigned)
        if total_remaining_demand > sum(remaining_capacity_by_drone):
            return False

        # Every individual customer must fit in at least one remaining payload
        # capacity. This is a cheap but useful feasibility test.
        for customer in unassigned:
            if all(customer.demand > capacity for capacity in remaining_capacity_by_drone):
                return False

        return True

    def _lower_bound(
        self,
        routes: tuple[tuple[Customer, ...], ...],
        next_customer_index: int,
    ) -> float:
        """Compute an admissible lower bound for a partial solution.

        The bound is:

        current route energy
        + a minimum incoming base-energy term for every unassigned customer.

        Each unassigned customer must eventually have one incoming edge in the
        final routes. Payload-dependent energy is nonnegative, so using only the
        base energy gives an optimistic bound.
        """

        current_energy = self._routes_objective(routes)
        unassigned = self.ordered_customers[next_customer_index:]
        optimistic_future_energy = sum(
            self.config.base_energy_rate * self._minimum_incoming_distance(customer)
            for customer in unassigned
        )
        return current_energy + optimistic_future_energy

    def _minimum_incoming_distance(self, customer: Customer) -> float:
        """Shortest possible incoming distance from any other node."""

        candidates: list[Location] = [self.instance.depot]
        candidates.extend(
            other for other in self.instance.customers if other.id != customer.id
        )
        return min(location.distance_to(customer) for location in candidates)

    def _routes_objective(self, routes: tuple[tuple[Customer, ...], ...]) -> float:
        """Compute total energy of a tuple-based route state."""

        total = 0.0
        for drone, customers in zip(self.instance.drones, routes):
            route = Route(drone=drone, depot=self.instance.depot, customers=customers)
            total += route.total_energy(
                base_rate=self.config.base_energy_rate,
                payload_rate=self.config.payload_energy_rate,
            )
        return total

    def _build_solution(
        self,
        routes: tuple[tuple[Customer, ...], ...],
        include_empty_routes: bool,
    ) -> Solution:
        """Convert the internal route state to the public ``Solution`` object."""

        solution_routes: list[Route] = []
        for drone, customers in zip(self.instance.drones, routes):
            if include_empty_routes or customers:
                solution_routes.append(
                    Route(
                        drone=drone,
                        depot=self.instance.depot,
                        customers=customers,
                    )
                )

        return Solution(instance=self.instance, routes=tuple(solution_routes))

    def _initialize_incumbent_with_greedy_solution(self) -> None:
        """Build a quick feasible solution to improve early pruning.

        This is not required for exactness. It only gives the Branch and Bound
        search an initial upper bound, which can dramatically reduce the number
        of explored nodes.
        """

        routes = tuple(() for _ in self.instance.drones)

        for customer in self.ordered_customers:
            best_candidate: tuple[float, tuple[tuple[Customer, ...], ...]] | None = None

            for drone_index, route_customers in enumerate(routes):
                for position in range(len(route_customers) + 1):
                    candidate_route = (
                        route_customers[:position]
                        + (customer,)
                        + route_customers[position:]
                    )
                    candidate_routes = (
                        routes[:drone_index]
                        + (candidate_route,)
                        + routes[drone_index + 1 :]
                    )

                    if not self._partial_routes_are_feasible(candidate_routes):
                        continue

                    candidate_objective = self._routes_objective(candidate_routes)
                    if best_candidate is None or candidate_objective < best_candidate[0]:
                        best_candidate = (candidate_objective, candidate_routes)

            if best_candidate is None:
                return

            routes = best_candidate[1]

        solution = self._build_solution(routes, include_empty_routes=False)
        is_valid, _ = solution.validate_feasibility(
            base_rate=self.config.base_energy_rate,
            payload_rate=self.config.payload_energy_rate,
        )
        if is_valid and self._solution_avoids_no_fly_zones(solution):
            self.best_solution = solution
            self.best_objective = solution.total_energy(
                base_rate=self.config.base_energy_rate,
                payload_rate=self.config.payload_energy_rate,
            )

    def _solution_avoids_no_fly_zones(self, solution: Solution) -> bool:
        """Return True if every route segment avoids all circular no-fly zones."""

        if not self.instance.no_fly_zones:
            return True

        for route in solution.routes:
            sequence: tuple[Location, ...] = (
                (route.depot,)
                + tuple(route.customers)
                + (route.depot,)
            )
            for start, end in zip(sequence, sequence[1:]):
                if not self._segment_avoids_no_fly_zones(start, end):
                    return False

        return True

    def _segment_avoids_no_fly_zones(self, start: Location, end: Location) -> bool:
        """Check whether a straight segment intersects any no-fly circle."""

        return all(
            not _segment_intersects_circle(start, end, zone)
            for zone in self.instance.no_fly_zones
        )

    def _should_stop(self) -> bool:
        """Check optional runtime and node-count limits."""

        if self.config.max_nodes is not None and self.explored_nodes >= self.config.max_nodes:
            return True

        if self.config.time_limit_seconds is not None:
            elapsed = time.perf_counter() - self._start_time
            if elapsed >= self.config.time_limit_seconds:
                return True

        return False

    def _log_node(self, node: SearchNode, event: str) -> None:
        """Record search progress for later reporting."""

        important_event = event != "explored"
        frequency_hit = (
            self.config.log_frequency > 0
            and self.explored_nodes % self.config.log_frequency == 0
        )

        if important_event or frequency_hit or self.explored_nodes <= 5:
            self.logs.append(
                NodeLog(
                    explored_nodes=self.explored_nodes,
                    depth=node.depth,
                    current_objective=node.current_objective,
                    lower_bound=node.lower_bound,
                    best_objective=self.best_objective,
                    event=event,
                )
            )


def solve_branch_and_bound(
    instance: ProblemInstance,
    config: BranchAndBoundConfig | None = None,
) -> BranchAndBoundResult:
    """Convenience function for users and experiment scripts."""

    return BranchAndBoundSolver(instance=instance, config=config).solve()


def _segment_intersects_circle(
    start: Location,
    end: Location,
    zone: NoFlyZone,
) -> bool:
    """Return True if a line segment intersects a circular no-fly zone.

    The calculation projects the circle center onto the line segment and then
    measures the distance from the center to the closest point on the segment.
    If that distance is smaller than the radius, the straight drone movement
    would cross restricted airspace.
    """

    dx = end.x - start.x
    dy = end.y - start.y

    if dx == 0 and dy == 0:
        return zone.contains(start)

    numerator = (zone.center_x - start.x) * dx + (zone.center_y - start.y) * dy
    denominator = dx * dx + dy * dy
    projection = max(0.0, min(1.0, numerator / denominator))

    closest_x = start.x + projection * dx
    closest_y = start.y + projection * dy
    distance_to_segment = math.hypot(zone.center_x - closest_x, zone.center_y - closest_y)

    return distance_to_segment <= zone.radius
