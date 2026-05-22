"""Visualize depot, customers, no-fly zones, and drone routes.

The functions in this module are intentionally lightweight and use matplotlib
only. They are designed for the academic report and experimental study: every
saved plot shows the benchmark geometry, the selected drone paths, and the
energy consumption values needed to compare optimization methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from drone_delivery.core.entities import Customer, ProblemInstance
from drone_delivery.core.solution import Route, Solution


DEFAULT_OUTPUT_DIR = Path("outputs") / "plots"


def plot_instance(
    instance: ProblemInstance,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    show_labels: bool = True,
    dpi: int = 150,
) -> Path:
    """Plot depot, customers, and no-fly zones without routes.

    This is useful for documenting benchmark instances before optimization.
    The figure is saved to ``outputs/plots`` by default.
    """

    output_path = _prepare_output_path(
        output_dir,
        file_name or f"{instance.instance_id}_instance.png",
    )
    figure, axes = _create_base_figure(instance)

    _plot_no_fly_zones(axes, instance)
    _plot_depot(axes, instance)
    _plot_customers(axes, instance.customers, show_labels=show_labels)
    _finalize_axes(
        axes,
        title=f"Drone Delivery Instance: {instance.instance_id}",
        instance=instance,
    )

    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_solution(
    solution: Solution,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    file_name: str | None = None,
    show_labels: bool = True,
    base_energy_rate: float = 1.0,
    payload_energy_rate: float = 0.03,
    dpi: int = 150,
) -> Path:
    """Plot a complete solution with one color per drone route.

    The plot includes:

    - depot marker,
    - customer markers and demands,
    - circular no-fly zones,
    - route polylines with arrows,
    - per-route energy in the legend,
    - total energy in the title.
    """

    instance = solution.instance
    output_path = _prepare_output_path(
        output_dir,
        file_name or f"{instance.instance_id}_solution.png",
    )
    figure, axes = _create_base_figure(instance)

    _plot_no_fly_zones(axes, instance)
    _plot_routes(
        axes,
        solution.routes,
        base_energy_rate=base_energy_rate,
        payload_energy_rate=payload_energy_rate,
    )
    _plot_depot(axes, instance)
    _plot_customers(axes, instance.customers, show_labels=show_labels)

    total_energy = solution.total_energy(
        base_rate=base_energy_rate,
        payload_rate=payload_energy_rate,
    )
    _add_energy_summary(
        axes,
        solution,
        base_energy_rate=base_energy_rate,
        payload_energy_rate=payload_energy_rate,
    )
    _finalize_axes(
        axes,
        title=(
            f"Drone Routes: {instance.instance_id} "
            f"(Total energy = {total_energy:.2f})"
        ),
        instance=instance,
    )

    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _create_base_figure(instance: ProblemInstance) -> tuple[Figure, Axes]:
    """Create the matplotlib figure and axes used by all route plots."""

    figure, axes = plt.subplots(figsize=(9, 7))
    axes.set_facecolor("#f8fafc")
    axes.grid(True, color="#d9e2ec", linewidth=0.7, alpha=0.75)

    # Use equal aspect ratio so Euclidean distances are visually meaningful.
    axes.set_aspect("equal", adjustable="box")
    return figure, axes


def _plot_depot(axes: Axes, instance: ProblemInstance) -> None:
    """Draw the depot as a large square marker."""

    depot = instance.depot
    axes.scatter(
        [depot.x],
        [depot.y],
        marker="s",
        s=160,
        color="#111827",
        edgecolor="white",
        linewidth=1.5,
        label="Depot",
        zorder=5,
    )
    axes.annotate(
        "Depot",
        (depot.x, depot.y),
        textcoords="offset points",
        xytext=(8, 8),
        fontsize=9,
        weight="bold",
    )


def _plot_customers(
    axes: Axes,
    customers: Iterable[Customer],
    show_labels: bool,
) -> None:
    """Draw customer points and optionally annotate id and demand."""

    customers = tuple(customers)
    if not customers:
        return

    axes.scatter(
        [customer.x for customer in customers],
        [customer.y for customer in customers],
        marker="o",
        s=70,
        color="#ffffff",
        edgecolor="#334155",
        linewidth=1.4,
        label="Customers",
        zorder=4,
    )

    if show_labels:
        for customer in customers:
            axes.annotate(
                f"C{customer.id}\nq={customer.demand:g}",
                (customer.x, customer.y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#0f172a",
            )


def _plot_no_fly_zones(axes: Axes, instance: ProblemInstance) -> None:
    """Draw circular no-fly zones as translucent red disks."""

    for index, zone in enumerate(instance.no_fly_zones):
        circle = Circle(
            (zone.center_x, zone.center_y),
            zone.radius,
            facecolor="#ef4444",
            edgecolor="#991b1b",
            linewidth=1.2,
            alpha=0.22,
            label="No-fly zone" if index == 0 else None,
            zorder=1,
        )
        axes.add_patch(circle)
        axes.annotate(
            f"NFZ {zone.id}",
            (zone.center_x, zone.center_y),
            ha="center",
            va="center",
            fontsize=8,
            color="#7f1d1d",
            zorder=2,
        )


def _plot_routes(
    axes: Axes,
    routes: tuple[Route, ...],
    base_energy_rate: float,
    payload_energy_rate: float,
) -> None:
    """Draw each drone route using a distinct color."""

    colors = plt.get_cmap("tab10")

    for index, route in enumerate(routes):
        if not route.customers:
            continue

        color = colors(index % 10)
        sequence = (route.depot,) + route.customers + (route.depot,)
        x_values = [location.x for location in sequence]
        y_values = [location.y for location in sequence]
        route_energy = route.total_energy(
            base_rate=base_energy_rate,
            payload_rate=payload_energy_rate,
        )

        axes.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4,
            label=(
                f"Drone {route.drone.id}: "
                f"E={route_energy:.2f}, D={route.total_distance():.2f}"
            ),
            zorder=3,
        )

        # Small arrows make route direction visible without needing a separate
        # animation or interactive plot.
        for start, end in zip(sequence, sequence[1:]):
            axes.annotate(
                "",
                xy=(end.x, end.y),
                xytext=(start.x, start.y),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": 1.4,
                    "shrinkA": 7,
                    "shrinkB": 7,
                    "alpha": 0.85,
                },
                zorder=3,
            )


def _add_energy_summary(
    axes: Axes,
    solution: Solution,
    base_energy_rate: float,
    payload_energy_rate: float,
) -> None:
    """Add a compact text box with total distance and energy."""

    total_distance = solution.total_distance()
    total_energy = solution.total_energy(
        base_rate=base_energy_rate,
        payload_rate=payload_energy_rate,
    )
    text = f"Total distance: {total_distance:.2f}\nTotal energy: {total_energy:.2f}"

    axes.text(
        0.02,
        0.98,
        text,
        transform=axes.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.92,
        },
        zorder=10,
    )


def _finalize_axes(axes: Axes, title: str, instance: ProblemInstance) -> None:
    """Apply labels, limits, title, and legend."""

    axes.set_title(title, fontsize=13, weight="bold")
    axes.set_xlabel("X coordinate")
    axes.set_ylabel("Y coordinate")

    x_values = [instance.depot.x] + [customer.x for customer in instance.customers]
    y_values = [instance.depot.y] + [customer.y for customer in instance.customers]
    for zone in instance.no_fly_zones:
        x_values.extend([zone.center_x - zone.radius, zone.center_x + zone.radius])
        y_values.extend([zone.center_y - zone.radius, zone.center_y + zone.radius])

    margin = 8.0
    axes.set_xlim(min(x_values) - margin, max(x_values) + margin)
    axes.set_ylim(min(y_values) - margin, max(y_values) + margin)
    axes.legend(loc="upper right", fontsize=8, framealpha=0.95)


def _prepare_output_path(output_dir: str | Path, file_name: str) -> Path:
    """Create the output directory and return a PNG output path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / file_name
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path
