"""Core domain entities for the drone delivery optimization problem.

These classes describe the static data of an instance: physical locations,
customer package demands, drone resource capacities, and restricted airspace.
They intentionally do not contain optimization logic. Solver-specific logic
will live in the algorithm modules, while route and solution behavior lives in
``solution.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any


@dataclass(frozen=True)
class Location:
    """A point in the two-dimensional Euclidean service area.

    The project uses Euclidean distances because benchmark customers are
    represented by Cartesian coordinates. This is common in academic routing
    experiments because it is simple, reproducible, and solver-independent.
    """

    id: int
    x: float
    y: float

    def distance_to(self, other: "Location") -> float:
        """Return the Euclidean distance to another location."""

        return math.hypot(self.x - other.x, self.y - other.y)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class Customer(Location):
    """Delivery customer with a package demand.

    The demand is the payload weight that must be carried by a drone when it
    leaves the depot. A feasible route cannot exceed the assigned drone's
    payload capacity.
    """

    demand: float


@dataclass(frozen=True)
class Depot(Location):
    """Central depot where every drone route starts and ends."""


@dataclass(frozen=True)
class Drone:
    """Drone resource limits.

    ``payload_capacity`` limits the total package demand assigned to the drone
    on one route. ``battery_capacity`` limits total energy consumed by that
    route.
    """

    id: int
    payload_capacity: float
    battery_capacity: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class NoFlyZone:
    """Circular restricted area.

    The first implementation stores no-fly zones as circles because they are
    easy to generate, visualize, and test for intersections. Later route
    feasibility can reject arcs that intersect these circles.
    """

    id: int
    center_x: float
    center_y: float
    radius: float

    def contains(self, location: Location, clearance: float = 0.0) -> bool:
        """Return True if a location lies inside the zone plus clearance."""

        distance = math.hypot(self.center_x - location.x, self.center_y - location.y)
        return distance <= self.radius + clearance

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a JSON-serializable dictionary."""

        return asdict(self)


@dataclass(frozen=True)
class ProblemInstance:
    """Complete input data for one drone delivery problem instance."""

    instance_id: str
    depot: Depot
    customers: tuple[Customer, ...]
    drones: tuple[Drone, ...]
    no_fly_zones: tuple[NoFlyZone, ...] = ()
    metadata: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None

    @property
    def customer_ids(self) -> set[int]:
        """Return the set of customers that must be served exactly once."""

        return {customer.id for customer in self.customers}

    @property
    def drone_ids(self) -> set[int]:
        """Return the set of available drone identifiers."""

        return {drone.id for drone in self.drones}

    def customer_by_id(self, customer_id: int) -> Customer:
        """Find a customer by id or raise a clear error."""

        for customer in self.customers:
            if customer.id == customer_id:
                return customer
        raise KeyError(f"Unknown customer id: {customer_id}")

    def drone_by_id(self, drone_id: int) -> Drone:
        """Find a drone by id or raise a clear error."""

        for drone in self.drones:
            if drone.id == drone_id:
                return drone
        raise KeyError(f"Unknown drone id: {drone_id}")

    def to_dict(self) -> dict[str, Any]:
        """Convert the full instance into the JSON schema used by the project."""

        return {
            "instance_id": self.instance_id,
            "metadata": self.metadata or {},
            "parameters": self.parameters or {},
            "depot": self.depot.to_dict(),
            "customers": [customer.to_dict() for customer in self.customers],
            "drones": [drone.to_dict() for drone in self.drones],
            "no_fly_zones": [zone.to_dict() for zone in self.no_fly_zones],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemInstance":
        """Build a ``ProblemInstance`` from the project JSON schema."""

        depot_data = data["depot"]
        return cls(
            instance_id=data["instance_id"],
            depot=Depot(
                id=depot_data.get("id", 0),
                x=depot_data["x"],
                y=depot_data["y"],
            ),
            customers=tuple(Customer(**customer) for customer in data["customers"]),
            drones=tuple(Drone(**drone) for drone in data["drones"]),
            no_fly_zones=tuple(
                NoFlyZone(**zone) for zone in data.get("no_fly_zones", [])
            ),
            metadata=data.get("metadata", {}),
            parameters=data.get("parameters", {}),
        )
