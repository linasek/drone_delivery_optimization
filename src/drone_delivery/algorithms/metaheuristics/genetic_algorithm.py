"""Genetic Algorithm adapted to drone route encoding and repair.

The GA is designed specifically for the drone delivery problem. A chromosome is
a tuple of drone routes, and custom operators from ``drone_delivery.operators``
perform crossover, mutation, and repair. This keeps the algorithm modular and
allows the same operators to be reused in later metaheuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

from drone_delivery.core.entities import Location, NoFlyZone, ProblemInstance
from drone_delivery.core.solution import Solution
from drone_delivery.operators import (
    RouteChromosome,
    chromosome_penalty,
    create_empty_chromosome,
    create_random_chromosome,
    mutate_chromosome,
    order_crossover,
    repair_chromosome,
)


@dataclass(frozen=True)
class GeneticAlgorithmConfig:
    """Configuration parameters controlling the GA search."""

    population_size: int = 60
    generations: int = 200
    tournament_size: int = 3
    crossover_rate: float = 0.85
    mutation_rate: float = 0.20
    elitism_count: int = 2
    no_improvement_generations: int = 40
    max_runtime_seconds: float | None = None
    random_seed: int = 42
    base_energy_rate: float = 1.0
    payload_energy_rate: float = 0.03
    penalty_weight: float = 10_000.0


@dataclass(frozen=True)
class EvaluatedChromosome:
    """Chromosome plus objective, penalty, and fitness values."""

    chromosome: RouteChromosome
    solution: Solution
    objective: float
    penalty: float
    penalized_cost: float
    fitness: float
    feasible: bool


@dataclass(frozen=True)
class GenerationLog:
    """Summary of one GA generation for reports and experiments."""

    generation: int
    best_objective: float
    best_penalized_cost: float
    average_penalized_cost: float
    feasible_count: int


@dataclass
class GeneticAlgorithmResult:
    """Result returned by the GA solver."""

    solution: Solution
    best_chromosome: RouteChromosome
    best_objective: float
    best_penalized_cost: float
    runtime_seconds: float
    generations_executed: int
    status: str
    logs: list[GenerationLog] = field(default_factory=list)

    @property
    def found_feasible_solution(self) -> bool:
        """Return True if the reported objective is for a feasible solution."""

        return math.isfinite(self.best_objective)


class GeneticAlgorithmSolver:
    """Genetic Algorithm solver for drone delivery optimization.

    Search cycle:
    1. Generate and repair an initial population.
    2. Evaluate fitness using energy plus constraint penalties.
    3. Preserve elite chromosomes.
    4. Select parents by tournament selection.
    5. Apply route-aware crossover.
    6. Apply route-aware mutation.
    7. Repair offspring and repeat until stopping criteria are met.
    """

    def __init__(
        self,
        instance: ProblemInstance,
        config: GeneticAlgorithmConfig | None = None,
    ) -> None:
        self.instance = instance
        self.config = config or GeneticAlgorithmConfig()
        self.rng = random.Random(self.config.random_seed)

        if self.config.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if self.config.elitism_count < 0:
            raise ValueError("elitism_count cannot be negative.")
        if self.config.elitism_count >= self.config.population_size:
            raise ValueError("elitism_count must be smaller than population_size.")

    def solve(self) -> GeneticAlgorithmResult:
        """Run the Genetic Algorithm."""

        start_time = time.perf_counter()
        population = self._initialize_population()
        best_feasible: EvaluatedChromosome | None = None
        best_penalized: EvaluatedChromosome | None = None
        logs: list[GenerationLog] = []
        generations_without_improvement = 0
        generations_executed = 0

        for generation in range(self.config.generations):
            if self._time_limit_reached(start_time):
                break

            evaluated = [self._evaluate(chromosome) for chromosome in population]
            evaluated.sort(key=lambda item: item.penalized_cost)

            current_best_penalized = evaluated[0]
            if (
                best_penalized is None
                or current_best_penalized.penalized_cost < best_penalized.penalized_cost
            ):
                best_penalized = current_best_penalized

            current_best_feasible = next(
                (item for item in evaluated if item.feasible),
                None,
            )
            if current_best_feasible is not None and (
                best_feasible is None
                or current_best_feasible.objective < best_feasible.objective
            ):
                best_feasible = current_best_feasible
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            logs.append(self._build_generation_log(generation, evaluated, best_feasible))
            generations_executed = generation + 1

            if generations_without_improvement >= self.config.no_improvement_generations:
                break

            population = self._create_next_generation(evaluated)

        runtime = time.perf_counter() - start_time
        selected = best_feasible or best_penalized
        if selected is None:
            # This defensive fallback should never be reached because the
            # population is nonempty, but it keeps the public API total.
            chromosome = repair_chromosome(
                create_empty_chromosome(self.instance),
                self.instance,
                self.rng,
                self.config.base_energy_rate,
                self.config.payload_energy_rate,
                self.config.penalty_weight,
            )
            selected = self._evaluate(chromosome)

        status = "feasible_solution_found" if best_feasible is not None else "no_feasible_solution_found"
        if self._time_limit_reached(start_time):
            status = f"{status}_time_limit"
        elif generations_executed < self.config.generations:
            status = f"{status}_stagnation"

        return GeneticAlgorithmResult(
            solution=selected.solution,
            best_chromosome=selected.chromosome,
            best_objective=best_feasible.objective if best_feasible else math.inf,
            best_penalized_cost=selected.penalized_cost,
            runtime_seconds=runtime,
            generations_executed=generations_executed,
            status=status,
            logs=logs,
        )

    def _initialize_population(self) -> list[RouteChromosome]:
        """Create the initial population and repair every chromosome."""

        population: list[RouteChromosome] = []

        # Seed one constructive chromosome by repairing an empty encoding. This
        # gives the population a deterministic baseline solution.
        population.append(
            repair_chromosome(
                create_empty_chromosome(self.instance),
                self.instance,
                self.rng,
                self.config.base_energy_rate,
                self.config.payload_energy_rate,
                self.config.penalty_weight,
            )
        )

        while len(population) < self.config.population_size:
            chromosome = create_random_chromosome(self.instance, self.rng)
            population.append(
                repair_chromosome(
                    chromosome,
                    self.instance,
                    self.rng,
                    self.config.base_energy_rate,
                    self.config.payload_energy_rate,
                    self.config.penalty_weight,
                )
            )

        return population

    def _evaluate(self, chromosome: RouteChromosome) -> EvaluatedChromosome:
        """Evaluate one chromosome using energy objective and penalties."""

        repaired = repair_chromosome(
            chromosome,
            self.instance,
            self.rng,
            self.config.base_energy_rate,
            self.config.payload_energy_rate,
            self.config.penalty_weight,
        )
        solution = repaired.to_solution(self.instance)
        objective = solution.total_energy(
            base_rate=self.config.base_energy_rate,
            payload_rate=self.config.payload_energy_rate,
        )

        penalty = chromosome_penalty(
            repaired,
            self.instance,
            self.config.base_energy_rate,
            self.config.payload_energy_rate,
            self.config.penalty_weight,
        )
        penalty += self._no_fly_zone_penalty(solution)

        penalized_cost = objective + penalty
        fitness = 1.0 / (1.0 + penalized_cost)
        feasible = penalty == 0.0

        return EvaluatedChromosome(
            chromosome=repaired,
            solution=solution,
            objective=objective,
            penalty=penalty,
            penalized_cost=penalized_cost,
            fitness=fitness,
            feasible=feasible,
        )

    def _create_next_generation(
        self,
        evaluated: list[EvaluatedChromosome],
    ) -> list[RouteChromosome]:
        """Create offspring population using elitism, selection, crossover, mutation."""

        next_population: list[RouteChromosome] = [
            item.chromosome for item in evaluated[: self.config.elitism_count]
        ]

        while len(next_population) < self.config.population_size:
            parent_a = self._tournament_selection(evaluated).chromosome
            parent_b = self._tournament_selection(evaluated).chromosome

            if self.rng.random() < self.config.crossover_rate:
                child = order_crossover(
                    parent_a,
                    parent_b,
                    self.instance,
                    self.rng,
                    self.config.base_energy_rate,
                    self.config.payload_energy_rate,
                    self.config.penalty_weight,
                )
            else:
                child = parent_a

            child = mutate_chromosome(
                child,
                self.instance,
                self.rng,
                self.config.mutation_rate,
                self.config.base_energy_rate,
                self.config.payload_energy_rate,
                self.config.penalty_weight,
            )
            next_population.append(child)

        return next_population

    def _tournament_selection(
        self,
        evaluated: list[EvaluatedChromosome],
    ) -> EvaluatedChromosome:
        """Select one parent using tournament selection."""

        tournament_size = min(self.config.tournament_size, len(evaluated))
        candidates = self.rng.sample(evaluated, tournament_size)
        return min(candidates, key=lambda item: item.penalized_cost)

    def _build_generation_log(
        self,
        generation: int,
        evaluated: list[EvaluatedChromosome],
        best_feasible: EvaluatedChromosome | None,
    ) -> GenerationLog:
        """Build a compact summary for one generation."""

        best_objective = best_feasible.objective if best_feasible else math.inf
        feasible_count = sum(1 for item in evaluated if item.feasible)
        average_cost = sum(item.penalized_cost for item in evaluated) / len(evaluated)

        return GenerationLog(
            generation=generation,
            best_objective=best_objective,
            best_penalized_cost=evaluated[0].penalized_cost,
            average_penalized_cost=average_cost,
            feasible_count=feasible_count,
        )

    def _no_fly_zone_penalty(self, solution: Solution) -> float:
        """Penalize route segments that intersect circular no-fly zones."""

        if not self.instance.no_fly_zones:
            return 0.0

        violations = 0
        for route in solution.routes:
            sequence: tuple[Location, ...] = (
                (route.depot,)
                + tuple(route.customers)
                + (route.depot,)
            )
            for start, end in zip(sequence, sequence[1:]):
                for zone in self.instance.no_fly_zones:
                    if _segment_intersects_circle(start, end, zone):
                        violations += 1

        return self.config.penalty_weight * violations

    def _time_limit_reached(self, start_time: float) -> bool:
        """Return True if the optional runtime limit has been reached."""

        if self.config.max_runtime_seconds is None:
            return False
        return time.perf_counter() - start_time >= self.config.max_runtime_seconds


def solve_genetic_algorithm(
    instance: ProblemInstance,
    config: GeneticAlgorithmConfig | None = None,
) -> GeneticAlgorithmResult:
    """Convenience function for experiment scripts and notebooks."""

    return GeneticAlgorithmSolver(instance=instance, config=config).solve()


def _segment_intersects_circle(
    start: Location,
    end: Location,
    zone: NoFlyZone,
) -> bool:
    """Return True if a straight route segment intersects a no-fly circle."""

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
