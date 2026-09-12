"""Calibrate one raster view from printed lengths and pixel measurements."""

import math
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

_ABSOLUTE_TOLERANCE_PX = 2.0
_RELATIVE_TOLERANCE = 0.02


def _fit_pixels_per_mm(
    nominal: list[float],
    measured: list[float],
    thresholds: list[float],
    inliers: tuple[int, ...],
) -> float:
    """Fit through the origin, weighting by the pixel error allowance.

    The printed length is the reference; measurement noise is on the pixel
    side. Normalise the design column before squaring to avoid overflow.
    """
    x = [nominal[i] / thresholds[i] for i in inliers]
    y = [measured[i] / thresholds[i] for i in inliers]
    reach = max(x)
    if reach == 0:
        return math.inf
    x = [value / reach for value in x]
    return math.fsum(a * b for a, b in zip(x, y)) / math.fsum(a * a for a in x) / reach


def _candidates(
    nominal: list[float],
    measured: list[float],
    thresholds: list[float],
) -> dict[tuple[int, ...], float]:
    """RANSAC with every one-measurement hypothesis instead of random draws.

    Refit and reclassify until membership is stable. Discard cycles rather
    than returning an inlier mask that disagrees with its fitted scale.
    """
    candidates = {}
    for length, pixels in zip(nominal, measured):
        factor = pixels / length
        seen = set()
        previous = None
        while math.isfinite(factor) and factor > 0:
            inliers = tuple(
                i
                for i, (n, p, tolerance) in enumerate(
                    zip(nominal, measured, thresholds)
                )
                if abs(p - factor * n) <= tolerance
            )
            if not inliers:
                break
            fitted = _fit_pixels_per_mm(nominal, measured, thresholds, inliers)
            if inliers in seen:
                # Revisiting the immediately preceding mask is convergence;
                # revisiting an earlier one is a cycle.
                if (
                    inliers == previous
                    and math.isfinite(1 / factor)
                    and all(math.isfinite(factor * n) for n in nominal)
                ):
                    candidates[inliers] = factor
                break
            seen.add(inliers)
            previous = inliers
            factor = fitted
    return candidates


class _Measurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        pattern=r"^dim_[a-z0-9_]+$",
        description=(
            "Stable dim_ name for this printed dimension, in lower_snake_case. "
            "Assign it now if no Dimension has been submitted yet; reuse it "
            "when submitting Dimension.name and when revising this reading. "
            "Each printed dimension may appear only once per call."
        ),
    )
    nominal: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Printed length converted to millimetres, not an angle or a "
            "feature count. Must describe the same extent as measured: "
            "both diameter, both radius, or the same linear distance."
        ),
    )
    measured: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Corresponding length in pixels, measured on the exact raster "
            "file that will be named by DrawingView.file, after any crop "
            "or resize. Do not pair a diameter with a radius, or a "
            "horizontal distance with a slanted distance."
        ),
    )


def calculate_drawing_scale(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate measurement pairs and fit mm/pixel without invoking a tool."""
    readings = [_Measurement.model_validate(item) for item in measurements]
    names = [m.name for m in readings]
    nominal = [m.nominal for m in readings]
    measured = [m.measured for m in readings]
    thresholds = [_ABSOLUTE_TOLERANCE_PX + _RELATIVE_TOLERANCE * p for p in measured]

    def result(
        status: str,
        message: str,
        factor: float | None = None,
        inliers: tuple[int, ...] = (),
        alternatives: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "scale": None if factor is None else 1 / factor,
            "unit": "mm/px",
            "message": message,
            "threshold": {
                "absolute_px": _ABSOLUTE_TOLERANCE_PX,
                "relative": _RELATIVE_TOLERANCE,
                "rule": "absolute_px + relative * measured",
            },
            "inliers": [names[i] for i in inliers],
            "outliers": [
                name
                for i, name in enumerate(names)
                if factor is not None and i not in inliers
            ],
            "measurements": [
                {
                    **m.model_dump(),
                    "residual_px": (
                        None if factor is None else m.measured - factor * m.nominal
                    ),
                    "threshold_px": thresholds[i],
                    "inlier": None if factor is None else i in inliers,
                }
                for i, m in enumerate(readings)
            ],
            "alternatives": alternatives or [],
        }

    if len(set(names)) != len(names):
        return result(
            "invalid_input",
            "Names must be unique. Supply each printed dimension only once.",
        )
    if not readings:
        return result("insufficient_evidence", "Supply at least one printed length.")
    if any(
        not math.isfinite(n / p) or n / p == 0 or not math.isfinite(p / n) or p / n == 0
        for n, p in zip(nominal, measured)
    ):
        return result("invalid_input", "Length ratios exceed the numeric range.")
    if len(readings) == 1:
        return result(
            "insufficient_evidence",
            "Provisional scale from one dimension; no outlier check is possible.",
            measured[0] / nominal[0],
            (0,),
        )

    candidates = _candidates(nominal, measured, thresholds)
    if not candidates:
        return result("no_consensus", "No stable fit. Check the pixel measurements.")
    support = max(map(len, candidates))
    best = sorted(
        (
            (indices, factor)
            for indices, factor in candidates.items()
            if len(indices) == support
        ),
        key=lambda item: item[1],
    )
    alternatives = [
        {"scale": 1 / factor, "inliers": [names[i] for i in indices]}
        for indices, factor in best
    ]
    if len(best) > 1:
        return result(
            "ambiguous",
            "Different inlier sets have equal support. Check readings or add "
            "independent dimensions; no scale has been selected.",
            alternatives=alternatives,
        )
    if support * 2 <= len(readings):
        return result(
            "no_consensus",
            "The largest consensus is not a strict majority. Check for mixed "
            "views, incorrect readings, or nonuniform scale.",
            alternatives=alternatives,
        )
    indices, factor = best[0]
    return result(
        "ok",
        "Multiply pixel lengths by scale to obtain millimetres.",
        factor,
        indices,
    )


def create_calculate_drawing_scale_tool() -> BaseTool:
    @tool("calculate_drawing_scale")
    def _calculate_drawing_scale_tool(
        measurements: list[_Measurement],
    ) -> dict[str, Any]:
        """Fit millimetres per pixel for one raster view whose scale is uniform.

        Give the printed lengths and the pixel lengths you measured for them,
        all off the same file at the same resolution. The result is the sheet's
        `scale`: length_mm = length_px * scale. It converts lengths only, never
        origins or axes.

        A null scale means the pairs did not agree well enough to choose one;
        `status` and `alternatives` say how. Fix the readings rather than pick
        one.
        """
        return calculate_drawing_scale([item.model_dump() for item in measurements])

    return _calculate_drawing_scale_tool
