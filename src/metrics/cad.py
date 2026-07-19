"""Aggregate CadQuery execution and validity metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Mapping


def _row_mapping(row: object) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    if is_dataclass(row) and not isinstance(row, type):
        return asdict(row)
    raise TypeError(
        f"metric rows must be mappings or dataclass instances, got {type(row)!r}"
    )


def cad_execution_metrics(
    rows: Iterable[object],
    *,
    prefix: str = "val",
) -> dict[str, float | int]:
    """Compute execution/result/validity rates over all requested samples."""

    materialized = [_row_mapping(row) for row in rows]
    total = len(materialized)

    def rate(key: str) -> float:
        if total == 0:
            return 0.0
        return float(sum(bool(row.get(key, False)) for row in materialized) / total)

    stem = f"{prefix}/" if prefix else ""
    return {
        f"{stem}num_samples": total,
        f"{stem}exec_ok_rate": rate("exec_ok"),
        f"{stem}has_result_rate": rate("has_result"),
        f"{stem}valid_rate": rate("valid"),
    }


def cad_error_histogram(rows: Iterable[object]) -> dict[str, int]:
    """Return a deterministic error histogram suitable for artifact logging."""

    errors: Counter[str] = Counter()
    for item in rows:
        row = _row_mapping(item)
        raw = row.get("error")
        label = "ok" if raw in {None, ""} else str(raw).split(":", maxsplit=1)[0]
        errors[label] += 1
    return dict(sorted(errors.items(), key=lambda item: (-item[1], item[0])))


__all__ = ["cad_error_histogram", "cad_execution_metrics"]
