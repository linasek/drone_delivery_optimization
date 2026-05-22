"""Load and save instances, solutions, and experiment results."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    """Save a JSON object with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def load_instance(path: str | Path) -> dict[str, Any]:
    """Load one drone delivery benchmark instance."""

    return load_json(path)


def save_instance(instance: dict[str, Any], path: str | Path) -> Path:
    """Save one drone delivery benchmark instance."""

    return save_json(instance, path)
