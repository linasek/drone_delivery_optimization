"""Route and solution objects for drone delivery optimization.

The classes in this module represent candidate decisions. They are shared by
the exact solver and metaheuristics so every method evaluates feasibility and
energy in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from drone_delivery.core.energy import route_energy_from_segments
from drone_delivery.core.entities import Customer, Depot, Drone, ProblemInstance


@dataclass(frozen=True)
class Route:
    """Ordered delivery route assigned to one drone.

    A route is continuous by construction: it starts at the depot, visits its
    ordered customer list, and returns to the same depot. Empty routes are
    allowed because an available drone may remain unused.
    """

    drone: Drone
    depot: Depot
    customers: tuple[Customer, ...]

    @property
    def customer_ids(self) -> list[int]:
        """Customer identifiers in visit order."""

        return [customer.id for customer in self.customers]

    def total_payload(self) -> float:
        """Return the total demand carried when the drone leaves the depot."""

        return sum(customer.demand for customer in self.customers)

    def segment_distances(self) -> list[float]:
        """Return distances for all consecutive route segments.

        For customers [1, 2], the segments are depot->1, 1->2, and 2->depot.
        For an empty route, the distance is zero because the drone does not fly.
        """

        if not self.customers:
            return [0.0]

        distances: list[float] = []
        previous = self.depot

        for customer in self.customers:
            distances.append(previous.distance_to(customer))
            previous = customer

        distances.append(previous.distance_to(self.depot))
        return distances

    def total_distance(self) -> float:
        """Return the complete route length including return to the depot."""

        return sum(self.segment_distances())

    def total_energy(
        self,
        base_rate: float = 1.0,
        payload_rate: float = 0.03,
    ) -> float:
        """Return total route energy under the shared payload-sensitive model."""

        demands = [customer.demand for customer in self.customers]
        return route_energy_from_segments(
            self.segment_distances(),
            demands,
            base_rate=base_rate,
            payload_rate=payload_rate,
        )

    def is_payload_feasible(self) -> bool:
        """Check whether assigned demand fits the drone payload capacity."""

        return self.total_payload() <= self.drone.payload_capacity

    def is_battery_feasible(
        self,
        base_rate: float = 1.0,
        payload_rate: float = 0.03,
    ) -> bool:
        """Check whether route energy fits the drone battery capacity."""

        return self.total_energy(base_rate, payload_rate) <= self.drone.battery_capacity

    def is_feasible(
        self,
        base_rate: float = 1.0,
        payload_rate: float = 0.03,
    ) -> bool:
        """Check route-level payload and battery feasibility."""

        return self.is_payload_feasible() and self.is_battery_feasible(
            base_rate,
            payload_rate,
        )


@dataclass(frozen=True)
class Solution:
    """Candidate solution containing one route per used drone."""

    instance: ProblemInstance
    routes: tuple[Route, ...]

    def total_distance(self) -> float:
        """Return the total distance traveled by all routes."""

        return sum(route.total_distance() for route in self.routes)

    def total_energy(
        self,
        base_rate: float = 1.0,
        payload_rate: float = 0.03,
    ) -> float:
        """Return the optimization objective value for the full solution."""

        return sum(
            route.total_energy(base_rate=base_rate, payload_rate=payload_rate)
            for route in self.routes
        )

    def assigned_customer_ids(self) -> list[int]:
        """Return all assigned customers, preserving route order."""

        return [
            customer.id
            for route in self.routes
            for customer in route.customers
        ]

    def validate_customer_assignment(self) -> tuple[bool, list[str]]:
        """Verify that each instance customer is served exactly once.

        The method returns a boolean and a list of messages instead of raising
        exceptions. This is convenient for algorithms: a metaheuristic can use
        the messages for repair or penalties, while tests can assert that the
        list is empty.
        """

        messages: list[str] = []
        required = self.instance.customer_ids
        assigned = self.assigned_customer_ids()
        assigned_set = set(assigned)

        missing = sorted(required - assigned_set)
        if missing:
            messages.append(f"Missing customers: {missing}")

        unknown = sorted(assigned_set - required)
        if unknown:
            messages.append(f"Unknown customers: {unknown}")

        duplicates = sorted(
            customer_id for customer_id in assigned_set if assigned.count(customer_id) > 1
        )
        if duplicates:
            messages.append(f"Duplicated customers: {duplicates}")

        return not messages, messages

    def validate_drone_assignment(self) -> tuple[bool, list[str]]:
        """Verify that every route uses an available drone at most once."""

        messages: list[str] = []
        available_drones = self.instance.drone_ids
        used_drones = [route.drone.id for route in self.routes]
        used_set = set(used_drones)

        unknown = sorted(used_set - available_drones)
        if unknown:
            messages.append(f"Unknown drones: {unknown}")

        duplicated = sorted(
            drone_id for drone_id in used_set if used_drones.count(drone_id) > 1
        )
        if duplicated:
            messages.append(f"Drones assigned to multiple routes: {duplicated}")

        return not messages, messages

    def validate_feasibility(
        self,
        base_rate: float = 1.0,
        payload_rate: float = 0.03,
    ) -> tuple[bool, list[str]]:
        """Validate assignment, payload, and battery constraints."""

        messages: list[str] = []

        for validator in (
            self.validate_customer_assignment,
            self.validate_drone_assignment,
        ):
            _, validator_messages = validator()
            messages.extend(validator_messages)

        for route in self.routes:
            if not route.is_payload_feasible():
                messages.append(
                    f"Drone {route.drone.id} exceeds payload capacity: "
                    f"{route.total_payload():.3f} > {route.drone.payload_capacity:.3f}"
                )

            if not route.is_battery_feasible(base_rate, payload_rate):
                messages.append(
                    f"Drone {route.drone.id} exceeds battery capacity: "
                    f"{route.total_energy(base_rate, payload_rate):.3f} > "
                    f"{route.drone.battery_capacity:.3f}"
                )

        return not messages, messages
