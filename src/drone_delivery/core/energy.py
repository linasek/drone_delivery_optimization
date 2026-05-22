"""Energy consumption models for drone routes.

The project objective is to minimize total energy consumption, not merely
distance. We therefore use a simple academic energy model that increases with
both traveled distance and currently carried payload.
"""

from __future__ import annotations


def arc_energy(
    distance: float,
    carried_payload: float,
    base_rate: float = 1.0,
    payload_rate: float = 0.03,
) -> float:
    """Return energy consumed on one route segment.

    Parameters
    ----------
    distance:
        Euclidean length of the segment.
    carried_payload:
        Payload carried while flying over this segment. On a delivery route,
        this decreases after each customer is served.
    base_rate:
        Energy per distance unit for an empty drone.
    payload_rate:
        Additional energy per distance and payload unit.

    The formula is:
        energy = distance * (base_rate + payload_rate * carried_payload)
    """

    if distance < 0:
        raise ValueError("Distance cannot be negative.")
    if carried_payload < -1e-9:
        raise ValueError("Carried payload cannot be negative.")
    carried_payload = max(0.0, carried_payload)

    return distance * (base_rate + payload_rate * carried_payload)


def route_energy_from_segments(
    segment_distances: list[float],
    delivered_demands: list[float],
    base_rate: float = 1.0,
    payload_rate: float = 0.03,
) -> float:
    """Compute route energy from segment distances and delivery sequence.

    ``segment_distances`` has one more item than ``delivered_demands`` because
    it includes depot-to-first-customer, customer-to-customer arcs, and the
    final return to the depot.
    """

    if len(segment_distances) != len(delivered_demands) + 1:
        raise ValueError("A route with n customers must have n + 1 segments.")

    carried_payload = sum(delivered_demands)
    total_energy = 0.0

    for index, distance in enumerate(segment_distances):
        total_energy += arc_energy(distance, carried_payload, base_rate, payload_rate)
        if index < len(delivered_demands):
            carried_payload -= delivered_demands[index]

    return total_energy
