"""Translation of the store-agnostic metadata filter dialect.

Supported forms::

    {"module": "payroll"}                  equality
    {"module": ["payroll", "leave"]}       membership
    {"fiscal_year": {"$gte": 2025}}        explicit operator

Backends that cannot express a predicate natively (FAISS) use
:func:`build_predicate` to post-filter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_service.core.exceptions import VectorStoreError

MetadataFilter = dict[str, Any]

OPERATORS = frozenset({"$eq", "$ne", "$in", "$nin", "$gt", "$gte", "$lt", "$lte", "$contains"})


def _compare(value: Any, operator: str, expected: Any) -> bool:
    match operator:
        case "$eq":
            return bool(value == expected)
        case "$ne":
            return bool(value != expected)
        case "$in":
            return value in expected
        case "$nin":
            return value not in expected
        case "$gt":
            return value is not None and value > expected
        case "$gte":
            return value is not None and value >= expected
        case "$lt":
            return value is not None and value < expected
        case "$lte":
            return value is not None and value <= expected
        case "$contains":
            return str(expected) in str(value)
        case _:
            raise VectorStoreError(f"Unsupported filter operator {operator!r}")


def _normalise(filters: MetadataFilter) -> list[tuple[str, dict[str, Any]]]:
    clauses: list[tuple[str, dict[str, Any]]] = []
    for key, expected in filters.items():
        if isinstance(expected, dict):
            unknown = set(expected) - OPERATORS
            if unknown:
                raise VectorStoreError(
                    f"Unsupported filter operator(s) {sorted(unknown)!r} for field {key!r}"
                )
            clauses.append((key, dict(expected)))
        elif isinstance(expected, (list, tuple, set)):
            clauses.append((key, {"$in": list(expected)}))
        else:
            clauses.append((key, {"$eq": expected}))
    return clauses


def build_predicate(filters: MetadataFilter | None) -> Callable[[dict[str, Any]], bool] | None:
    """Compile ``filters`` into a python metadata predicate (FAISS post-filter)."""
    if not filters:
        return None
    clauses = _normalise(filters)

    def _predicate(metadata: dict[str, Any]) -> bool:
        return all(
            all(
                _compare(metadata.get(key), operator, expected)
                for operator, expected in ops.items()
            )
            for key, ops in clauses
        )

    return _predicate


def to_chroma_where(filters: MetadataFilter | None) -> dict[str, Any] | None:
    """Render a Chroma ``where`` clause (multiple fields need an explicit ``$and``)."""
    if not filters:
        return None
    clauses = [{key: ops} for key, ops in _normalise(filters)]
    for clause in clauses:
        for ops in clause.values():
            if "$contains" in ops:
                raise VectorStoreError("Chroma metadata filters do not support $contains")
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def to_pinecone_filter(filters: MetadataFilter | None) -> dict[str, Any] | None:
    """Render a Pinecone metadata filter (top level keys are implicitly ANDed)."""
    if not filters:
        return None
    result: dict[str, Any] = {}
    for key, ops in _normalise(filters):
        if "$contains" in ops:
            raise VectorStoreError("Pinecone metadata filters do not support $contains")
        result[key] = ops
    return result
