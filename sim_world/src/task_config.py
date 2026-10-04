"""Load the scene and observation settings shared by demos and collectors."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Union
import re

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent
_FRACTION_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*/\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)


def load_task_config(config: Union[str, Path]) -> Dict[str, Any]:
    """Load one complete task configuration and validate its required sections."""
    path = Path(config)
    if not path.is_absolute():
        path = path if path.is_file() else _CONFIG_DIR / path
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    if not isinstance(data, Mapping):
        raise ValueError(f"Task config must be a mapping: {path}")
    required_mappings = ("make_scene", "collect_observation")
    missing = [key for key in required_mappings if not isinstance(data.get(key), Mapping)]
    if missing:
        raise ValueError(f"Missing mapping(s) {missing} in task config: {path}")
    return deepcopy(dict(data))


def parse_number(value: Any, *, field: str) -> float:
    """Parse a YAML number or a simple fraction such as ``1 / 120``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        match = _FRACTION_PATTERN.fullmatch(value)
        if match:
            numerator, denominator = map(float, match.groups())
            if denominator == 0.0:
                raise ValueError(f"{field} denominator must not be zero")
            return numerator / denominator
    raise ValueError(f"{field} must be a number or fraction like '1 / 120': {value!r}")
