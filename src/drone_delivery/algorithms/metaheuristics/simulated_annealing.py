"""Simulated Annealing solver using drone-routing neighborhood moves.

Simulated Annealing (SA) is a single-solution metaheuristic. It starts from one
candidate solution and repeatedly moves to a neighboring solution. Improving
moves are always accepted, while worsening moves may be accepted with a
probability that decreases as the temperature cools. This controlled
randomness helps the search escape local optima.
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
    create_random_chromosome,
    generate_neighbor,
    repair_chromosome,
)


@dataclass(frozen=True)
class SimulatedAnnealingConfig:
    """Configuration parameters for the SA search."""

    initial_temperature: float = 500.0
    minimum_temperature: float = 1e-3
    cooling_rate: float = 0.95
    iterations_per_temperature: int = 50
    max_iterations: int = 10_000
    no_improvement_iterations: int = 2_000
    max_runtime_seconds: float | None = None
    random_seed: int = 42
    base_energy_rate: float = 1.0
    payload_energy_rate: float = 0.03
    penalty_weight: float = 10_000.0


@dataclass(frozen=True)
class AnnealingLog:
    """Search trace entry used for reports and plots."""

    iteration: int
    temperature: float
    current_cost: float
    best_penalized_cost: float
    best_objective: float
    accepted: bool
    event: str


@dataclass(frozen=True)
class EvaluatedState:
    """A chromosome with its objective, penalty, and feasibility status."""

    chromosome: RouteChromosome
    solution: Solution
    objective: float
    penalty: float
    penalized_cost: float
    feasible: bool


@dataclass
class SimulatedAnnealingResult:
    """Result returned by the SA solver."""

    solution: Solution
    best_chromosome: RouteChromosome
    best_objective: float
    best_penalized_cost: float
    runtime_seconds: float
    iterations_executed: int
    final_temperature: float
    status: str
    logs: list[AnnealingLog] = field(default_factory=list)

    @property
    def found_feasible_solution(self) -> bool:
        """Return True if the best objective corresponds to a feasible route set."""

        return math.isfinite(self.best_objective)


class SimulatedAnnealingSolver:
    """Simulated Annealing solver for drone delivery optimization."""

    def __init__(
        self,
        instance: ProblemInstance,
        config: SimulatedAnnealingConfig | None = None,
    ) -> None:
        self.instance = instance
        self.config = config or SimulatedAnnealingConfig()
        self.rng = random.Random(self.config.random_seed)

        if self.config.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive.")
        if not 0 < self.config.cooling_rate < 1:
            raise ValueError("cooling_rate must be between 0 and 1.")
        if self.config.iterations_per_temperature <= 0:
            raise ValueError("iterations_per_temperature must be positive.")

    def solve(self) -> SimulatedAnnealingResult:
        """Run the Simulated Annealing search."""

        start_time = time.perf_counter()
        current = self._evaluate(self._initial_chromosome())
        best_penalized = current
        best_feasible = current if current.feasible else None
        logs: list[AnnealingLog] = []

        temperature = self.config.initial_temperature
        iteration = 0
        iterations_without_improvement = 0

        while self._continue_search(
            temperature,
            iteration,
            iterations_without_improvement,
            start_time,
        ):
            for _ in range(self.config.iterations_per_temperature):
                if not self._continue_search(
                    temperature,
                    iteration,
                    iterations_without_improvement,
                    start_time,
                ):
                    break

                neighbor = self._evaluate(
                    generate_neighbor(
                        current.chromosome,
                        self.instance,
                        self.rng,
                        self.config.base_energy_rate,
                        self.config.payload_energy_rate,
                        self.config.penalty_weight,
                    )
                )

                accepted = self._accept_neighbor(current, neighbor, temperature)
                event = "rejected"

                if accepted:
                    accepted_worse = neighbor.penalized_cost > current.penalized_cost
                    current = neighbor
                    event = "accepted_worse" if accepted_worse else "accepted"

                improved = False
                if current.penalized_cost < best_penalized.penalized_cost:
                    best_penalized = current
                    improved = True
                    event = "new_best_penalized"

                if current.feasible and (
                    best_feasible is None or current.objective < best_feasible.objective
                ):
                    best_feasible = current
                    improved = True
                    event = "new_best_feasible"

                if improved:
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1

                iteration += 1
                self._append_log(
                    logs,
                    iteration,
                    temperature,
                    current,
                    best_penalized,
                    best_feasible,
                    accepted,
                    event,
                )

            temperature *= self.config.cooling_rate

        runtime = time.perf_counter() - start_time
        selected = best_feasible or best_penalized

        status = "feasible_solution_found" if best_feasible else "no_feasible_solution_found"
        if self._time_limit_reached(start_time):
            status = f"{status}_time_limit"
        elif temperature <= self.config.minimum_temperature:
            status = f"{status}_cooled"
        elif iteration >= self.config.max_iterations:
            status = f"{status}_max_iterations"
        elif iterations_without_improvement >= self.config.no_improvement_iterations:
            status = f"{status}_stagnation"

        return SimulatedAnnealingResult(
            solution=selected.solution,
            best_chromosome=selected.chromosome,
            best_objective=best_feasible.objective if best_feasible else math.inf,
            best_penalized_cost=selected.penalized_cost,
            runtime_seconds=runtime,
            iterations_executed=iteration,
            final_temperature=temperature,
            status=status,
            logs=logs,
        )

    def _initial_chromosome(self) -> RouteChromosome:
        """Create and repair an initial route chromosome."""

        chromosome = create_random_chromosome(self.instance, self.rng)
        return repair_chromosome(
            chromosome,
            self.instance,
            self.rng,
            self.config.base_energy_rate,
            self.config.payload_energy_rate,
            self.config.penalty_weight,
        )

    def _evaluate(self, chromosome: RouteChromosome) -> EvaluatedState:
        """Evaluate objective, penalty, and feasibility for one state."""

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

        return EvaluatedState(
            chromosome=repaired,
            solution=solution,
            objective=objective,
            penalty=penalty,
            penalized_cost=objective + penalty,
            feasible=penalty == 0.0,
        )

    def _accept_neighbor(
        self,
        current: EvaluatedState,
        neighbor: EvaluatedState,
        temperature: float,
    ) -> bool:
        """Apply the Metropolis acceptance rule."""

        delta = neighbor.penalized_cost - current.penalized_cost
        if delta <= 0:
            return True

        probability = math.exp(-delta / temperature)
        return self.rng.random() < probability

    def _continue_search(
        self,
        temperature: float,
        iteration: int,
        iterations_without_improvement: int,
        start_time: float,
    ) -> bool:
        """Check all stopping criteria."""

        if temperature <= self.config.minimum_temperature:
            return False
        if iteration >= self.config.max_iterations:
            return False
        if iterations_without_improvement >= self.config.no_improvement_iterations:
            return False
        if self._time_limit_reached(start_time):
            return False
        return True

    def _append_log(
        self,
        logs: list[AnnealingLog],
        iteration: int,
        temperature: float,
        current: EvaluatedState,
        best_penalized: EvaluatedState,
        best_feasible: EvaluatedState | None,
        accepted: bool,
        event: str,
    ) -> None:
        """Record compact progress information.

        Logging every iteration is useful for small academic experiments, but
        the condition keeps logs readable for longer runs.
        """

        important = event.startswith("new_best") or iteration <= 5
        periodic = iteration % self.config.iterations_per_temperature == 0
        if not important and not periodic:
            return

        logs.append(
            AnnealingLog(
                iteration=iteration,
                temperature=temperature,
                current_cost=current.penalized_cost,
                best_penalized_cost=best_penalized.penalized_cost,
                best_objective=best_feasible.objective if best_feasible else math.inf,
                accepted=accepted,
                event=event,
            )
        )

    def _no_fly_zone_penalty(self, solution: Solution) -> float:
        """Penalize straight segments that intersect no-fly zones."""

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


def solve_simulated_annealing(
    instance: ProblemInstance,
    config: SimulatedAnnealingConfig | None = None,
) -> SimulatedAnnealingResult:
    """Convenience function for scripts, notebooks, and experiments."""

    return SimulatedAnnealingSolver(instance=instance, config=config).solve()


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
