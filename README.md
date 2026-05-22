# Drone-Based Delivery Optimization Problem

Professional Python project for optimizing drone delivery routes from a depot to customers while minimizing total energy consumption.

This repository is organized so the mathematical report, exact method, metaheuristics, benchmark generation, experiments, and visualizations can be developed step by step without mixing concerns.

## Project Structure

```text
drone_delivery_optimization/
├── data/
│   ├── raw/                         # External or manually provided datasets.
│   ├── processed/                   # Cleaned/normalized datasets ready for algorithms.
│   └── benchmarks/                  # Generated benchmark instances used in experiments.
├── docs/
│   ├── formulations/                # MILP and graph/flow-based mathematical formulations.
│   ├── complexity/                  # NP-hardness and complexity study notes.
│   └── report_notes/                # Reusable explanations, tables, and writing material.
├── experiments/
│   ├── configs/                     # Experiment parameter files.
│   └── run_experiments.py           # Main script for benchmark comparisons.
├── notebooks/                       # Optional exploratory analysis notebooks.
├── outputs/
│   ├── plots/                       # Runtime, quality, and scalability plots.
│   ├── routes/                      # Route visualization images.
│   └── tables/                      # CSV/Excel result tables.
├── scripts/                         # Command-line helper scripts.
├── src/drone_delivery/
│   ├── algorithms/
│   │   ├── exact/                   # Branch and Bound solver.
│   │   └── metaheuristics/          # Genetic Algorithm and Simulated Annealing.
│   ├── core/                        # Shared domain entities and feasibility logic.
│   ├── data/                        # Dataset generation, loading, and saving.
│   ├── evaluation/                  # Metrics and comparative analysis utilities.
│   ├── models/                      # Mathematical model descriptions/builders.
│   ├── operators/                   # Encoding, crossover, mutation, neighborhood, repair.
│   ├── utils/                       # Config, random seeds, logging, geometry helpers.
│   └── visualization/               # Route and experiment plotting.
└── tests/
    ├── unit/                        # Focused tests for individual modules.
    └── integration/                 # End-to-end tests for solvers and experiments.
```

## Development Roadmap

1. Write the two distinct mathematical formulations.
2. Add the complexity study and report-ready explanations.
3. Implement the shared domain model and feasibility checks.
4. Implement the Branch and Bound exact method.
5. Implement Genetic Algorithm and Simulated Annealing.
6. Generate at least 10 benchmark instances.
7. Run comparative experiments and create result tables/plots.
8. Write the final comparative analysis.

