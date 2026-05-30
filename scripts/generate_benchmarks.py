"""Command-line entry point for generating benchmark instances.

Example:
    python scripts/generate_benchmarks.py --count 10 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from drone_delivery.data.generator import generate_benchmark_set


def parse_args() -> argparse.Namespace:
    """Parse command-line options for benchmark generation."""

    parser = argparse.ArgumentParser(
        description="Generate random drone delivery benchmark instances.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of benchmark instances to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Instance i uses seed + i - 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "benchmarks",
        help="Directory where JSON benchmark files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate benchmarks and print the saved file paths."""

    args = parse_args()
    saved_files = generate_benchmark_set(
        output_dir=args.output_dir,
        count=args.count,
        base_seed=args.seed,
    )

    print(f"Generated {len(saved_files)} benchmark instances:")
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
