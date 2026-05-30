from math import isclose, isinf

from drone_delivery.algorithms.exact import BranchAndBoundConfig, solve_branch_and_bound
from drone_delivery.core import Customer, Depot, Drone, ProblemInstance


def test_branch_and_bound_solves_small_distance_instance():
    instance = ProblemInstance(
        instance_id="small_exact",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=1.0),
            Customer(id=2, x=2.0, y=0.0, demand=1.0),
        ),
        drones=(Drone(id=1, payload_capacity=5.0, battery_capacity=20.0),),
    )

    result = solve_branch_and_bound(
        instance,
        BranchAndBoundConfig(base_energy_rate=1.0, payload_energy_rate=0.0),
    )

    assert result.status == "optimal"
    assert result.found_solution
    assert isclose(result.best_objective, 4.0)
    assert result.explored_nodes > 0
    assert result.runtime_seconds >= 0.0


def test_branch_and_bound_reports_infeasible_payload_instance():
    instance = ProblemInstance(
        instance_id="payload_infeasible",
        depot=Depot(id=0, x=0.0, y=0.0),
        customers=(
            Customer(id=1, x=1.0, y=0.0, demand=10.0),
            Customer(id=2, x=2.0, y=0.0, demand=10.0),
        ),
        drones=(Drone(id=1, payload_capacity=5.0, battery_capacity=100.0),),
    )

    result = solve_branch_and_bound(instance)

    assert result.status == "infeasible"
    assert not result.found_solution
    assert isinf(result.best_objective)
    assert result.pruned_by_infeasibility > 0
