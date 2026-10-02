import math
import multiprocessing as mp
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from zeroshot.pipeline.verification.drawing_diff.align import (
    AlignmentResult,
    Backend,
    Model,
)


@dataclass(frozen=True)
class DrawingDiffReport:
    drawing_path: Path
    projection_path: Path | None
    alignment: AlignmentResult | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    paths: dict[str, Path] = field(default_factory=dict)
    error: str | None = None


class DrawingDiffExecutor:
    def __init__(
        self,
        *,
        backend: Backend = "directional_chamfer",
        model: Model = "similarity",
        alignment_options: dict[str, Any] | None = None,
        distance_clip_px: float | None = 12.0,
        timeout_seconds: float = 90.0,
    ) -> None:
        if backend not in ("directional_chamfer", "match_anything"):
            raise ValueError(f"invalid backend: {backend}")
        if model not in ("similarity", "affine", "homography"):
            raise ValueError(f"invalid model: {model}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if distance_clip_px is not None and distance_clip_px <= 0:
            raise ValueError("distance_clip_px must be None or positive")
        options = dict(alignment_options or {})
        if not 0 < options.pop("scale_tolerance", 0.01) < 1:
            raise ValueError("scale_tolerance must be in (0, 1)")

        if backend == "directional_chamfer":
            from .drawing_diff.directional_chamfer import validate_options
        else:
            from .drawing_diff.match_anything import validate_options
        validate_options(model, options)

        self.backend = backend
        self.model = model
        self.alignment_options = alignment_options or {}
        self.distance_clip_px = distance_clip_px
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        pairs: Sequence[tuple[Path, Path]],
        *,
        drawing_scales: Sequence[float | None] | None = None,
    ) -> list[DrawingDiffReport]:
        """Compare pairs; drawing_scales are input millimetres per native pixel."""
        scales = (
            list(drawing_scales) if drawing_scales is not None else [None] * len(pairs)
        )
        if len(scales) != len(pairs) or any(
            scale is not None and (not math.isfinite(scale) or scale <= 0)
            for scale in scales
        ):
            raise ValueError(
                "drawing_scales must contain one positive finite scale or None per pair"
            )
        if not pairs:
            return []

        deadline = time.monotonic() + self.timeout_seconds
        process, receiver = self._start(pairs, scales)
        try:
            return self._collect(process, receiver, pairs, deadline)
        finally:
            if process.is_alive():
                process.kill()
            process.join()
            process.close()
            receiver.close()

    def _start(
        self, pairs: Sequence[tuple[Path, Path]], drawing_scales: Sequence[float | None]
    ) -> tuple[BaseProcess, Connection]:
        # lazy import to prevent circular dependency
        from .drawing_diff.worker import run_worker

        context = mp.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = None
        try:
            process = context.Process(
                target=run_worker,
                args=(pairs, sender),
                kwargs={
                    "backend": self.backend,
                    "model": self.model,
                    "alignment_options": self.alignment_options,
                    "distance_clip_px": self.distance_clip_px,
                    "drawing_scales": drawing_scales,
                },
            )
            process.start()
            return process, receiver
        except BaseException:
            # retrieve falures before handing back to execute()
            if process is not None:
                if process.pid is not None:
                    if process.is_alive():
                        process.kill()
                    process.join()
                process.close()
            receiver.close()
            raise
        finally:
            # If the parent holds the sender, the child's exit cannot be detected by EOF.
            sender.close()

    def _collect(
        self,
        process: BaseProcess,
        receiver: Connection,
        pairs: Sequence[tuple[Path, Path]],
        deadline: float,
    ) -> list[DrawingDiffReport]:
        count = len(pairs)
        reports: list[DrawingDiffReport | None] = [None] * count

        # receive before join in order to prevent the child from blocking the pipe.
        for _ in range(count):
            if not receiver.poll(max(0.0, deadline - time.monotonic())):
                break
            try:
                index, report = receiver.recv()
            except EOFError as error:
                raise RuntimeError(
                    "Drawing-diff worker exited before all results arrived"
                ) from error

            if (
                type(index) is not int
                or not 0 <= index < count
                or reports[index] is not None
                or not isinstance(report, DrawingDiffReport)
            ):
                raise RuntimeError("Invalid drawing-diff worker result")

            reports[index] = report

        # wait for the child to exit within the remaining time, and check for abnormal exit.
        process.join(max(0.0, deadline - time.monotonic()))
        if process.exitcode not in (None, 0):
            raise RuntimeError(
                f"Drawing-diff worker exited with code {process.exitcode}"
            )
        if any(report is None for report in reports) and not process.is_alive():
            raise RuntimeError("Drawing-diff worker exited without all results")

        # Leave the received results even if the worker timed out.
        return [
            report
            if report is not None
            else DrawingDiffReport(
                drawing_path=drawing_path,
                projection_path=projection_path,
                error=f"Drawing-diff timed out after {self.timeout_seconds:g}s",
            )
            for report, (drawing_path, projection_path) in zip(
                reports, pairs, strict=True
            )
        ]
