"""Objective value, runtime, feasibility, and solution quality metrics.

This module converts solver outputs into a common academic evaluation format.
The exact method, Genetic Algorithm, and Simulated Annealing all report
different internal statistics, but the experimental study needs one consistent
table containing runtime, objective value, feasibility, and convergence data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import math
from pathlib import Path
import time
from typing import Any, Callable, TypeVar

from drone_delivery.core.entities import Location, NoFlyZone
from drone_delivery.core.solution import Solution


T = TypeVar("T")


@dataclass(frozen=True)
class FeasibilityReport:
    """Feasibility status and diagnostic messages for one solution."""

    feasible: bool
    messages: tuple[str, ...]


@dataclass(frozen=True)
class AlgorithmMetrics:
    """Comparable metrics for one algorithm on one instance."""

    instance_id: str
    algorithm: str
    status: str
    feasible: bool
    objective_value: float
    penalized_cost: float
    runtime_seconds: float
    total_distance: float
    explored_nodes: int | None = None
    iterations: int | None = None
    generations: int | None = None
    pruned_by_bound: int | None = None
    pruned_by_infeasibility: int | None = None
    completed_solutions: int | None = None
    feasibility_messages: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        """Convert metrics into a flat row suitable for CSV export."""

        row = asdict(self)
        row["objective_value"] = _finite_or_empty(self.objective_value)
        row["penalized_cost"] = _finite_or_empty(self.penalized_cost)
        row["total_distance"] = _finite_or_empty(self.total_distance)
        return row


@dataclass(frozen=True)
class ConvergencePoint:
    """One convergence-tracking observation from a solver log."""

    instance_id: str
    algorithm: str
    step: int
    best_objective: float
    best_penalized_cost: float
    current_cost: float | None = None
    lower_bound: float | None = None
    temperature: float | None = None
    event: str = ""

    def to_csv_row(self) -> dict[str, Any]:
        """Convert convergence point into a flat CSV row."""

        row = asdict(self)
        for key in ("best_objective", "best_penalized_cost", "current_cost", "lower_bound", "temperature"):
            row[key] = _finite_or_empty(row[key])
        return row


def measure_runtime(function: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float]:
    """Execute a callable and return its output plus measured runtime.

    Some solvers already measure runtime internally. This helper is still useful
    for future functions or external baselines that do not expose timing.
    """

    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    runtime_seconds = time.perf_counter() - start_time
    return result, runtime_seconds


def evaluate_solution_feasibility(
    solution: Solution | None,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    enforce_no_fly_zones: bool = True,
) -> FeasibilityReport:
    """Check customer assignment, payload, battery, and no-fly-zone feasibility."""

    if solution is None:
        return FeasibilityReport(False, ("No solution returned.",))

    is_valid, messages = solution.validate_feasibility(
        base_rate=base_energy_rate,
        payload_rate=payload_energy_rate,
    )

    if enforce_no_fly_zones:
        no_fly_messages = _no_fly_zone_messages(solution)
        messages.extend(no_fly_messages)
        is_valid = is_valid and not no_fly_messages

    return FeasibilityReport(is_valid, tuple(messages))


def solution_objective(
    solution: Solution | None,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
) -> float:
    """Return solution energy, or infinity if no solution exists."""

    if solution is None:
        return math.inf
    return solution.total_energy(base_rate=base_energy_rate, payload_rate=payload_energy_rate)


def solution_distance(solution: Solution | None) -> float:
    """Return solution distance, or infinity if no solution exists."""

    if solution is None:
        return math.inf
    return solution.total_distance()


def export_metrics_to_csv(
    metrics: list[AlgorithmMetrics],
    output_path: str | Path,
) -> Path:
    """Export comparison metrics to a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(AlgorithmMetrics.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow(row.to_csv_row())

    return path


def export_convergence_to_csv(
    convergence: list[ConvergencePoint],
    output_path: str | Path,
) -> Path:
    """Export convergence history to a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(ConvergencePoint.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for point in convergence:
            writer.writerow(point.to_csv_row())

    return path


def _no_fly_zone_messages(solution: Solution) -> list[str]:
    """Return violation messages for route segments crossing no-fly zones."""

    messages: list[str] = []
    zones = solution.instance.no_fly_zones
    if not zones:
        return messages

    for route in solution.routes:
        sequence: tuple[Location, ...] = (route.depot,) + route.customers + (route.depot,)
        for start, end in zip(sequence, sequence[1:]):
            for zone in zones:
                if _segment_intersects_circle(start, end, zone):
                    messages.append(
                        "Drone "
                        f"{route.drone.id} segment {start.id}->{end.id} "
                        f"intersects no-fly zone {zone.id}."
                    )

    return messages


def _segment_intersects_circle(start: Location, end: Location, zone: NoFlyZone) -> bool:
    """Return True when a straight route segment intersects a circular zone."""

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


def _finite_or_empty(value: float | int | None) -> float | int | str | None:
    """Return empty string for infinite float values to keep CSV readable."""

    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value
