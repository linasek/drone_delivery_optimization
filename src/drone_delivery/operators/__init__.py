"""Custom solution operators for drone routing."""

from drone_delivery.operators.crossover import order_crossover, route_exchange_crossover
from drone_delivery.operators.encoding import (
    RouteChromosome,
    build_chromosome_from_sequence,
    create_empty_chromosome,
    create_random_chromosome,
    flatten_chromosome,
    solution_to_chromosome,
)
from drone_delivery.operators.mutation import (
    insertion_mutation,
    merge_mutation,
    mutate_chromosome,
    split_mutation,
    swap_mutation,
)
from drone_delivery.operators.neighborhood import (
    generate_neighbor,
    insertion_operator,
    route_merge_operator,
    route_split_operator,
    swap_operator,
)
from drone_delivery.operators.repair import (
    chromosome_penalty,
    is_chromosome_feasible,
    repair_chromosome,
    repair_infeasible_chromosome,
)

__all__ = [
    "RouteChromosome",
    "build_chromosome_from_sequence",
    "chromosome_penalty",
    "create_empty_chromosome",
    "create_random_chromosome",
    "flatten_chromosome",
    "generate_neighbor",
    "insertion_mutation",
    "insertion_operator",
    "is_chromosome_feasible",
    "merge_mutation",
    "mutate_chromosome",
    "order_crossover",
    "repair_infeasible_chromosome",
    "repair_chromosome",
    "route_exchange_crossover",
    "route_merge_operator",
    "route_split_operator",
    "solution_to_chromosome",
    "split_mutation",
    "swap_mutation",
    "swap_operator",
]
