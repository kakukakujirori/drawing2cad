"""Child process for drawing comparison: align pairs and save diff images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from PIL import Image

from ..run_drawing_diff import DrawingDiffReport
from .align import Backend, Model, align, read_rgb
from .diff import compute_diff


def run_worker(
    pairs: Sequence[tuple[Path, Path]],
    connection: Connection,
    *,
    backend: Backend = "directional_chamfer",
    model: Model = "similarity",
    alignment_options: Mapping[str, Any] | None = None,
    distance_clip_px: float | None = 12.0,
) -> None:
    """Initialize a runtime once and process the supplied image pairs in order."""
    try:
        runtime = None
        if backend == "match_anything":
            from .match_anything import load_runtime

            runtime = load_runtime()

        for idx, (drawing_path, projection_path) in enumerate(pairs):
            try:
                report = run_align_diff_save(
                    drawing_path=drawing_path,
                    projection_path=projection_path,
                    backend=backend,
                    model=model,
                    alignment_options=alignment_options,
                    distance_clip_px=distance_clip_px,
                    runtime=runtime,
                )
            except Exception as error:  # noqa: BLE001
                report = DrawingDiffReport(
                    drawing_path=drawing_path,
                    projection_path=projection_path,
                    error=f"{type(error).__name__}: {error}",
                )

            connection.send((idx, report))

    finally:
        connection.close()


def run_align_diff_save(
    drawing_path: Path,
    projection_path: Path,
    *,
    backend: Backend = "directional_chamfer",
    model: Model = "similarity",
    alignment_options: Mapping[str, Any] | None = None,
    distance_clip_px: float | None = 12.0,
    runtime: Any = None,
) -> DrawingDiffReport:
    """Align one pair, save available diff images, and return their paths."""

    drawing = read_rgb(drawing_path)
    projection = read_rgb(projection_path)

    alignment = align(
        drawing,
        projection,
        backend=backend,
        model=model,
        options=alignment_options,
        runtime=runtime,
    )
    diff = compute_diff(
        drawing,
        projection,
        alignment,
        distance_clip_px=distance_clip_px,
    )

    paths = {}
    if diff.overlay is not None:
        path = projection_path.with_name(f"{projection_path.stem}_overlay.png")
        Image.fromarray(diff.overlay).save(path)
        paths["overlay_path"] = path
    if diff.residual is not None:
        path = projection_path.with_name(f"{projection_path.stem}_residual.png")
        Image.fromarray(diff.residual).save(path)
        paths["residual_path"] = path

    return DrawingDiffReport(
        drawing_path=drawing_path,
        projection_path=projection_path,
        alignment=alignment,
        stats=diff.stats,
        warnings=diff.warnings,
        paths=paths,
    )
