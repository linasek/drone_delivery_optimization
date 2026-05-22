from drone_delivery.core import Customer, Depot, Drone, ProblemInstance, Route, Solution


def build_small_instance() -> ProblemInstance:
    depot = Depot(id=0, x=0.0, y=0.0)
    customers = (
        Customer(id=1, x=3.0, y=4.0, demand=5.0),
        Customer(id=2, x=6.0, y=8.0, demand=3.0),
    )
    drones = (
        Drone(id=1, payload_capacity=10.0, battery_capacity=100.0),
        Drone(id=2, payload_capacity=10.0, battery_capacity=100.0),
    )
    return ProblemInstance(
        instance_id="small",
        depot=depot,
        customers=customers,
        drones=drones,
    )


def test_location_distance_uses_euclidean_metric():
    instance = build_small_instance()

    assert instance.depot.distance_to(instance.customers[0]) == 5.0


def test_route_distance_and_energy_account_for_return_to_depot():
    instance = build_small_instance()
    route = Route(
        drone=instance.drones[0],
        depot=instance.depot,
        customers=instance.customers,
    )

    assert route.total_distance() == 20.0
    assert round(route.total_energy(), 3) == 21.65


def test_payload_and_battery_feasibility_checks():
    instance = build_small_instance()
    low_payload_drone = Drone(id=3, payload_capacity=4.0, battery_capacity=100.0)
    low_battery_drone = Drone(id=4, payload_capacity=10.0, battery_capacity=10.0)

    overloaded_route = Route(
        drone=low_payload_drone,
        depot=instance.depot,
        customers=instance.customers,
    )
    low_battery_route = Route(
        drone=low_battery_drone,
        depot=instance.depot,
        customers=instance.customers,
    )

    assert not overloaded_route.is_payload_feasible()
    assert not low_battery_route.is_battery_feasible()


def test_solution_validates_customer_assignment_exactly_once():
    instance = build_small_instance()
    valid_solution = Solution(
        instance=instance,
        routes=(
            Route(instance.drones[0], instance.depot, (instance.customers[0],)),
            Route(instance.drones[1], instance.depot, (instance.customers[1],)),
        ),
    )
    invalid_solution = Solution(
        instance=instance,
        routes=(
            Route(instance.drones[0], instance.depot, (instance.customers[0],)),
            Route(instance.drones[1], instance.depot, (instance.customers[0],)),
        ),
    )

    assert valid_solution.validate_customer_assignment() == (True, [])

    is_valid, messages = invalid_solution.validate_customer_assignment()
    assert not is_valid
    assert "Missing customers: [2]" in messages
    assert "Duplicated customers: [1]" in messages
