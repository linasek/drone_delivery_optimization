"""Feasibility helper functions.

These functions wrap the methods on ``Route`` and ``Solution``. They are useful
for algorithms that prefer a functional interface, while the object-oriented
methods remain the primary source of truth.
"""

from __future__ import annotations

from drone_delivery.core.solution import Route, Solution


def is_payload_feasible(route: Route) -> bool:
    """Return True if a route respects its drone payload capacity."""

    return route.is_payload_feasible()


def is_battery_feasible(
    route: Route,
    base_rate: float = 1.0,
    payload_rate: float = 0.03,
) -> bool:
    """Return True if a route respects its drone battery capacity."""

    return route.is_battery_feasible(base_rate, payload_rate)


def is_route_feasible(
    route: Route,
    base_rate: float = 1.0,
    payload_rate: float = 0.03,
) -> bool:
    """Return True if route-level payload and battery constraints hold."""

    return route.is_feasible(base_rate, payload_rate)


def validate_customer_assignment(solution: Solution) -> tuple[bool, list[str]]:
    """Check that every customer is served exactly once."""

    return solution.validate_customer_assignment()


def validate_solution_feasibility(
    solution: Solution,
    base_rate: float = 1.0,
    payload_rate: float = 0.03,
) -> tuple[bool, list[str]]:
    """Check all currently implemented solution constraints."""

    return solution.validate_feasibility(base_rate, payload_rate)
