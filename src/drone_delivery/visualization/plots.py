"""Plot experiment tables, convergence curves, and scalability charts.

The experimental study needs visual comparisons of runtime, objective value,
feasibility, and convergence. These functions use matplotlib only and save PNG
files automatically to ``outputs/plots`` by default.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_delivery.evaluation.metrics import AlgorithmMetrics, ConvergencePoint
from drone_delivery.visualization.routes import plot_instance, plot_solution


DEFAULT_PLOT_DIR = Path("outputs") / "plots"


def plot_runtime_comparison(
    metrics: list[AlgorithmMetrics],
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    file_name: str = "runtime_comparison.png",
) -> Path:
    """Create a grouped bar chart comparing algorithm runtimes."""

    return _plot_metric_grouped_bar(
        metrics=metrics,
        value_getter=lambda metric: metric.runtime_seconds,
        ylabel="Runtime (seconds)",
        title="Runtime Comparison",
        output_dir=output_dir,
        file_name=file_name,
    )


def plot_objective_comparison(
    metrics: list[AlgorithmMetrics],
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    file_name: str = "objective_comparison.png",
) -> Path:
    """Create a grouped bar chart comparing feasible objective values."""

    return _plot_metric_grouped_bar(
        metrics=metrics,
        value_getter=lambda metric: metric.objective_value,
        ylabel="Objective value (total energy)",
        title="Solution Quality Comparison",
        output_dir=output_dir,
        file_name=file_name,
    )


def plot_feasibility_summary(
    metrics: list[AlgorithmMetrics],
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    file_name: str = "feasibility_summary.png",
) -> Path:
    """Plot the number of feasible solutions found by each algorithm."""

    output_path = _prepare_output_path(output_dir, file_name)
    counts: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)

    for metric in metrics:
        totals[metric.algorithm] += 1
        if metric.feasible:
            counts[metric.algorithm] += 1

    algorithms = sorted(totals)
    feasible_counts = [counts[algorithm] for algorithm in algorithms]
    total_counts = [totals[algorithm] for algorithm in algorithms]

    figure, axes = plt.subplots(figsize=(8, 5))
    bars = axes.bar(algorithms, feasible_counts, color="#2563eb", label="Feasible")
    axes.bar(
        algorithms,
        total_counts,
        color="#cbd5e1",
        alpha=0.35,
        label="Total runs",
    )

    for bar, feasible, total in zip(bars, feasible_counts, total_counts):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{feasible}/{total}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes.set_title("Feasibility Summary", weight="bold")
    axes.set_ylabel("Number of instances")
    axes.set_ylim(0, max(total_counts) + 1 if total_counts else 1)
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    figure.autofmt_xdate(rotation=15)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_convergence_curves(
    convergence: list[ConvergencePoint],
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    max_instances: int | None = None,
) -> list[Path]:
    """Create convergence plots, one PNG per benchmark instance."""

    output_paths: list[Path] = []
    by_instance: dict[str, list[ConvergencePoint]] = defaultdict(list)
    for point in convergence:
        by_instance[point.instance_id].append(point)

    instance_ids = sorted(by_instance)
    if max_instances is not None:
        instance_ids = instance_ids[:max_instances]

    for instance_id in instance_ids:
        output_path = _prepare_output_path(
            output_dir,
            f"convergence_{instance_id}.png",
        )
        figure, axes = plt.subplots(figsize=(9, 5))

        by_algorithm: dict[str, list[ConvergencePoint]] = defaultdict(list)
        for point in by_instance[instance_id]:
            by_algorithm[point.algorithm].append(point)

        for algorithm, points in sorted(by_algorithm.items()):
            points = sorted(points, key=lambda point: point.step)
            x_values = [point.step for point in points]
            y_values = [
                _finite_or_none(point.best_penalized_cost)
                for point in points
            ]
            axes.plot(x_values, y_values, marker="o", linewidth=1.8, label=algorithm)

        axes.set_title(f"Convergence History: {instance_id}", weight="bold")
        axes.set_xlabel("Search step")
        axes.set_ylabel("Best penalized cost")
        axes.grid(alpha=0.25)
        axes.legend()
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)

    return output_paths


def generate_comparison_plots(
    metrics: list[AlgorithmMetrics],
    convergence: list[ConvergencePoint],
    output_dir: str | Path = DEFAULT_PLOT_DIR,
    max_convergence_instances: int | None = None,
) -> list[Path]:
    """Generate all experiment comparison plots."""

    paths = [
        plot_runtime_comparison(metrics, output_dir),
        plot_objective_comparison(metrics, output_dir),
        plot_feasibility_summary(metrics, output_dir),
    ]
    paths.extend(
        plot_convergence_curves(
            convergence,
            output_dir,
            max_instances=max_convergence_instances,
        )
    )
    return paths


def _plot_metric_grouped_bar(
    metrics: list[AlgorithmMetrics],
    value_getter,
    ylabel: str,
    title: str,
    output_dir: str | Path,
    file_name: str,
) -> Path:
    """Shared grouped bar chart implementation for experiment metrics."""

    output_path = _prepare_output_path(output_dir, file_name)
    instance_ids = sorted({metric.instance_id for metric in metrics})
    algorithms = sorted({metric.algorithm for metric in metrics})
    metric_lookup = {
        (metric.instance_id, metric.algorithm): metric
        for metric in metrics
    }

    x_positions = list(range(len(instance_ids)))
    bar_width = 0.8 / max(1, len(algorithms))
    figure_width = max(9, len(instance_ids) * 0.7)
    figure, axes = plt.subplots(figsize=(figure_width, 5.5))

    for algorithm_index, algorithm in enumerate(algorithms):
        offsets = [
            position - 0.4 + bar_width / 2 + algorithm_index * bar_width
            for position in x_positions
        ]
        values = []
        for instance_id in instance_ids:
            metric = metric_lookup.get((instance_id, algorithm))
            value = value_getter(metric) if metric else math.nan
            values.append(_finite_or_nan(value))

        axes.bar(offsets, values, width=bar_width, label=algorithm)

    axes.set_title(title, weight="bold")
    axes.set_ylabel(ylabel)
    axes.set_xlabel("Benchmark instance")
    axes.set_xticks(x_positions)
    axes.set_xticklabels(instance_ids, rotation=45, ha="right")
    axes.grid(axis="y", alpha=0.25)
    axes.legend()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _prepare_output_path(output_dir: str | Path, file_name: str) -> Path:
    """Create output directory and return a PNG path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / file_name
    if path.suffix.lower() != ".png":
        path = path.with_suffix(".png")
    return path


def _finite_or_nan(value: float | int | None) -> float:
    """Convert missing or infinite values into NaN for matplotlib."""

    if value is None:
        return math.nan
    if isinstance(value, float) and not math.isfinite(value):
        return math.nan
    return float(value)


def _finite_or_none(value: float | int | None) -> float | None:
    """Return None for infinite values so matplotlib breaks the line cleanly."""

    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return float(value)


__all__ = [
    "generate_comparison_plots",
    "plot_convergence_curves",
    "plot_feasibility_summary",
    "plot_instance",
    "plot_objective_comparison",
    "plot_runtime_comparison",
    "plot_solution",
]
