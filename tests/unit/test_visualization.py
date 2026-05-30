from pathlib import Path

from drone_delivery.core import Customer, Depot, Drone, NoFlyZone, ProblemInstance, Route, Solution
from drone_delivery.visualization import plot_instance, plot_solution


def build_visual_instance() -> ProblemInstance:
    return ProblemInstance(
        instance_id="visual_test",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=1.0),
            Customer(id=2, x=2.0, y=1.0, demand=1.5),
            Customer(id=3, x=0.0, y=2.0, demand=1.0),
        ),
        drones=(
            Drone(id=1, payload_capacity=5.0, battery_capacity=100.0),
            Drone(id=2, payload_capacity=5.0, battery_capacity=100.0),
        ),
        no_fly_zones=(NoFlyZone(id=1, center_x=1.0, center_y=1.0, radius=0.25),),
    )


def test_visualization_saves_instance_and_solution_plots():
    instance = build_visual_instance()
    solution = Solution(
        instance=instance,
        routes=(
            Route(instance.drones[0], instance.depot, (instance.customers[0], instance.customers[1])),
            Route(instance.drones[1], instance.depot, (instance.customers[2],)),
        ),
    )
    output_dir = Path("outputs") / "plots"

    instance_path = plot_instance(instance, output_dir=output_dir, file_name="test_instance_plot.png")
    solution_path = plot_solution(solution, output_dir=output_dir, file_name="test_solution_plot.png")

    assert instance_path.exists()
    assert solution_path.exists()
    assert instance_path.stat().st_size > 0
    assert solution_path.stat().st_size > 0
