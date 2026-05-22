from drone_delivery.data.generator import InstanceConfig, generate_instance


def test_generate_instance_has_required_sections():
    config = InstanceConfig(
        instance_id="test_instance",
        random_seed=7,
        num_customers=5,
        num_drones=2,
        num_no_fly_zones=2,
    )

    instance = generate_instance(config)

    assert instance["instance_id"] == "test_instance"
    assert len(instance["customers"]) == 5
    assert len(instance["drones"]) == 2
    assert "depot" in instance
    assert "no_fly_zones" in instance


def test_generate_instance_is_reproducible_with_same_seed():
    config = InstanceConfig(
        instance_id="repeatable",
        random_seed=99,
        num_customers=4,
        num_drones=2,
        num_no_fly_zones=1,
    )

    first = generate_instance(config)
    second = generate_instance(config)

    assert first == second
