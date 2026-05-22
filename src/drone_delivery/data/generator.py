"""Random benchmark instance generator.

The generator creates complete drone delivery instances that can be used by
the exact method, metaheuristics, visualizations, and experimental study.

Each JSON instance contains:
- one depot coordinate,
- a list of customers with coordinates and package demands,
- a list of drones with payload and battery capacities,
- circular no-fly zones,
- generation metadata for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from typing import Any

from drone_delivery.core.entities import Customer, Depot, Drone, NoFlyZone


@dataclass(frozen=True)
class InstanceConfig:
    """Configuration controlling random instance generation."""

    instance_id: str
    random_seed: int
    num_customers: int
    num_drones: int
    num_no_fly_zones: int
    area_width: float = 100.0
    area_height: float = 100.0
    min_demand: float = 1.0
    max_demand: float = 8.0
    min_payload_capacity: float = 10.0
    max_payload_capacity: float = 25.0
    min_battery_capacity: float = 120.0
    max_battery_capacity: float = 260.0
    min_zone_radius: float = 5.0
    max_zone_radius: float = 14.0


def generate_instance(config: InstanceConfig) -> dict[str, Any]:
    """Generate one complete benchmark instance.

    The depot is placed at the center of the area. Customer coordinates are
    sampled uniformly, package demands are sampled independently, and drone
    capacities are sampled from configurable ranges. No-fly zones are rejected
    if they cover the depot or any customer, which keeps the instance usable
    for routing while still forcing paths to reason about restricted airspace.
    """

    rng = random.Random(config.random_seed)
    depot = Depot(
        id=0,
        x=round(config.area_width / 2.0, 3),
        y=round(config.area_height / 2.0, 3),
    )

    customers = _generate_customers(config, rng)
    drones = _generate_drones(config, rng)
    no_fly_zones = _generate_no_fly_zones(config, rng, depot, customers)

    return {
        "instance_id": config.instance_id,
        "metadata": {
            "random_seed": config.random_seed,
            "description": "Random drone delivery benchmark instance.",
            "coordinate_system": "2D Euclidean plane",
            "distance_unit": "abstract distance unit",
            "demand_unit": "abstract package weight unit",
            "battery_unit": "abstract energy unit",
        },
        "parameters": {
            "num_customers": config.num_customers,
            "num_drones": config.num_drones,
            "num_no_fly_zones": len(no_fly_zones),
            "area_width": config.area_width,
            "area_height": config.area_height,
        },
        "depot": depot.to_dict(),
        "customers": [customer.to_dict() for customer in customers],
        "drones": [drone.to_dict() for drone in drones],
        "no_fly_zones": [zone.to_dict() for zone in no_fly_zones],
    }


def generate_benchmark_set(
    output_dir: str | Path,
    count: int = 10,
    base_seed: int = 42,
) -> list[Path]:
    """Generate and save a set of benchmark instances.

    Instances gradually increase in size. This gives the experimental study a
    small scalability ladder: exact methods should work on the smallest cases,
    while GA and SA can be compared as the number of customers grows.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    for index in range(1, count + 1):
        config = InstanceConfig(
            instance_id=f"benchmark_{index:02d}",
            random_seed=base_seed + index - 1,
            num_customers=6 + index * 2,
            num_drones=2 + (index // 4),
            num_no_fly_zones=1 + (index % 4),
        )
        instance = generate_instance(config)
        file_path = output_path / f"{config.instance_id}.json"
        save_instance(instance, file_path)
        saved_files.append(file_path)

    return saved_files


def save_instance(instance: dict[str, Any], file_path: str | Path) -> Path:
    """Save one generated instance as pretty JSON."""

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(instance, indent=2), encoding="utf-8")
    return path


def _generate_customers(config: InstanceConfig, rng: random.Random) -> list[Customer]:
    """Sample customer coordinates and package demands."""

    customers: list[Customer] = []
    for customer_id in range(1, config.num_customers + 1):
        customers.append(
            Customer(
                id=customer_id,
                x=round(rng.uniform(0, config.area_width), 3),
                y=round(rng.uniform(0, config.area_height), 3),
                demand=round(rng.uniform(config.min_demand, config.max_demand), 3),
            )
        )
    return customers


def _generate_drones(config: InstanceConfig, rng: random.Random) -> list[Drone]:
    """Sample heterogeneous drone payload and battery capacities."""

    drones: list[Drone] = []
    for drone_id in range(1, config.num_drones + 1):
        drones.append(
            Drone(
                id=drone_id,
                payload_capacity=round(
                    rng.uniform(
                        config.min_payload_capacity,
                        config.max_payload_capacity,
                    ),
                    3,
                ),
                battery_capacity=round(
                    rng.uniform(
                        config.min_battery_capacity,
                        config.max_battery_capacity,
                    ),
                    3,
                ),
            )
        )
    return drones


def _generate_no_fly_zones(
    config: InstanceConfig,
    rng: random.Random,
    depot: Depot,
    customers: list[Customer],
) -> list[NoFlyZone]:
    """Generate circular no-fly zones that do not contain required nodes."""

    zones: list[NoFlyZone] = []
    max_attempts = config.num_no_fly_zones * 100
    attempts = 0

    while len(zones) < config.num_no_fly_zones and attempts < max_attempts:
        attempts += 1
        candidate = NoFlyZone(
            id=len(zones) + 1,
            center_x=round(rng.uniform(0, config.area_width), 3),
            center_y=round(rng.uniform(0, config.area_height), 3),
            radius=round(
                rng.uniform(config.min_zone_radius, config.max_zone_radius),
                3,
            ),
        )

        if _zone_is_acceptable(candidate, depot, customers, zones):
            zones.append(candidate)

    return zones


def _zone_is_acceptable(
    zone: NoFlyZone,
    depot: Depot,
    customers: list[Customer],
    existing_zones: list[NoFlyZone],
) -> bool:
    """Reject zones that cover important nodes or heavily overlap old zones."""

    clearance = 2.0
    if _distance(zone.center_x, zone.center_y, depot.x, depot.y) <= zone.radius + clearance:
        return False

    for customer in customers:
        if _distance(zone.center_x, zone.center_y, customer.x, customer.y) <= zone.radius + clearance:
            return False

    for existing in existing_zones:
        center_distance = _distance(
            zone.center_x,
            zone.center_y,
            existing.center_x,
            existing.center_y,
        )
        if center_distance <= 0.5 * (zone.radius + existing.radius):
            return False

    return True


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance helper used during generation."""

    return math.hypot(x1 - x2, y1 - y2)
