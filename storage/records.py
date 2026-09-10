"""Domain-neutral records and portable metadata predicates."""

from dataclasses import dataclass, field
import json
import math
from typing import Any


class StorageError(RuntimeError):
    pass


@dataclass
class DocumentRecord:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    content: str = ""


@dataclass
class VectorMatch:
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    content: str = ""
    metric: str = "cosine"


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def validate_filter(where):
    for key, expected in (where or {}).items():
        if not isinstance(key, str) or key.startswith("$"):
            raise ValueError("Filters must use metadata field names")
        if isinstance(expected, dict):
            if not expected or set(expected) - {"$eq", "$in", "$gte", "$lte"}:
                raise ValueError(f"Unsupported filter predicate for {key}")
            if "$in" in expected and not isinstance(expected["$in"], (list, tuple)):
                raise ValueError("$in expects a list")


def matches(metadata, where):
    validate_filter(where)
    for key, expected in (where or {}).items():
        if key not in metadata:
            return False
        actual = metadata[key]
        if not isinstance(expected, dict):
            if actual != expected:
                return False
            continue
        for op, value in expected.items():
            try:
                valid = {"$eq": lambda: actual == value,
                         "$in": lambda: actual in value,
                         "$gte": lambda: actual >= value,
                         "$lte": lambda: actual <= value}[op]()
            except TypeError:
                valid = False
            if not valid:
                return False
    return True


def validate_vector(vector, dimension=None):
    if not vector or (dimension is not None and len(vector) != dimension):
        raise ValueError("Vector dimension mismatch or empty vector")
    if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in vector):
        raise ValueError("Vectors must contain finite numbers")


def validate_limit(limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
