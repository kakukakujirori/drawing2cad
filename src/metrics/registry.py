"""Name-to-class registry backing the ``evaluation.metrics`` config list."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from src.metrics.base import ARTIFACTS, CADMetric


METRIC_REGISTRY: dict[str, type[CADMetric]] = {}

MetricT = TypeVar("MetricT", bound=type[CADMetric])


def register_metric(cls: MetricT) -> MetricT:
    """Register a metric family under its own class name."""

    name = cls.__name__
    existing = METRIC_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"metric name {name!r} is already registered")
    unknown = set(cls.requires) - ARTIFACTS
    if unknown:
        raise ValueError(
            f"metric {name!r} requires unknown artifact(s) {sorted(unknown)}; "
            f"known artifacts are {sorted(ARTIFACTS)}"
        )
    cls.name = name
    METRIC_REGISTRY[name] = cls
    return cls


def _known() -> str:
    return ", ".join(sorted(METRIC_REGISTRY)) or "<none registered>"


def _build_one(spec: Any) -> CADMetric:
    if isinstance(spec, CADMetric):
        return spec
    if isinstance(spec, str):
        name, kwargs = spec, {}
    elif isinstance(spec, Mapping):
        payload = dict(spec)
        raw_name = payload.pop("name", None)
        if raw_name is None:
            raise ValueError(
                f"metric mapping must carry a 'name' key, got keys {sorted(payload)}"
            )
        name, kwargs = str(raw_name), payload
    else:
        raise TypeError(
            "each metric entry must be a class name or a mapping with 'name', "
            f"got {type(spec)!r}"
        )
    cls = METRIC_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown metric {name!r}; known metrics are: {_known()}")
    try:
        return cls(**kwargs)
    except TypeError as error:
        raise TypeError(
            f"cannot build metric {name!r} from {kwargs!r}: {error}"
        ) from error


def build_metrics(specs: Sequence[Any] | None) -> tuple[CADMetric, ...]:
    """Instantiate the configured metric families, rejecting ambiguous sets.

    ``None`` selects nothing, which is how a caller disables scoring entirely.
    Duplicate families and colliding row keys are refused here rather than
    silently producing a row whose columns overwrite one another.
    """

    if specs is None:
        return ()
    if isinstance(specs, (str, Mapping)):
        raise TypeError("evaluation.metrics must be a list of metric entries")
    metrics = tuple(_build_one(spec) for spec in specs)

    seen_names: set[str] = set()
    owner_of_key: dict[str, str] = {}
    for metric in metrics:
        name = type(metric).__name__
        if name in seen_names:
            raise ValueError(f"metric {name!r} is listed more than once")
        seen_names.add(name)
        for key in metric.row_keys:
            owner = owner_of_key.get(key)
            if owner is not None:
                raise ValueError(
                    f"metrics {owner!r} and {name!r} both write row key {key!r}"
                )
            owner_of_key[key] = name
    return metrics


def required_artifacts(metrics: Sequence[CADMetric]) -> frozenset[str]:
    """Return the union of artifacts the selected metrics read."""

    required: set[str] = set()
    for metric in metrics:
        required.update(metric.requires)
    return frozenset(required)


def empty_metric_row(metrics: Sequence[CADMetric]) -> dict[str, Any]:
    """Return the row columns of a sample none of the metrics could score."""

    row: dict[str, Any] = {}
    for metric in metrics:
        row.update(metric.empty_row())
    return row


__all__ = [
    "METRIC_REGISTRY",
    "build_metrics",
    "empty_metric_row",
    "register_metric",
    "required_artifacts",
]
