from __future__ import annotations

import json
import math
from typing import Any


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number {token!r} is not permitted")


def _require_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path} is not permitted")
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _require_finite(child, f"{path}[{index}]")


def strict_json_loads(payload: str | bytes | bytearray) -> Any:
    """Decode RFC-compliant JSON and reject NaN, Infinity, and overflowed exponents."""

    value = json.loads(payload, parse_constant=_reject_constant)
    _require_finite(value)
    return value
