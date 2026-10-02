"""The drawing-diff executor keeps worker failures distinct from image failures."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zeroshot.pipeline.verification import run_drawing_diff
from zeroshot.pipeline.verification.drawing_diff import worker
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.run_drawing_diff import (
    DrawingDiffExecutor,
    DrawingDiffReport,
)


def _send_out_of_order(pairs, connection, **_options):
    try:
        for index in reversed(range(len(pairs))):
            connection.send(
                (
                    index,
                    DrawingDiffReport(
                        *pairs[index],
                        stats={
                            "index": index,
                            "scale": _options["drawing_scales"][index],
                            "tolerance": _options["alignment_options"][
                                "scale_tolerance"
                            ],
                        },
                    ),
                )
            )
    finally:
        connection.close()


def _send_one_then_hang(pairs, connection, **_options):
    connection.send((0, DrawingDiffReport(*pairs[0], stats={"completed": True})))
    time.sleep(60)


def _exit_early(_pairs, _connection, **_options):
    os._exit(3)


def _exit_after_sending(pairs, connection, **_options):
    for index in range(len(pairs)):
        connection.send((index, DrawingDiffReport(*pairs[index])))
    os._exit(3)


def _close_early(_pairs, connection, **_options):
    connection.close()


def _send_bad_index(pairs, connection, **_options):
    try:
        connection.send(("zero", DrawingDiffReport(*pairs[0])))
    finally:
        connection.close()


def _pairs(tmp_path: Path, count: int = 2) -> list[tuple[Path, Path]]:
    return [
        (tmp_path / f"drawing_{index}.png", tmp_path / f"projection_{index}.png")
        for index in range(count)
    ]


def test_empty_batch_does_not_start_a_worker(monkeypatch):
    def fail_if_started(_method):
        raise AssertionError("empty batch should not create a process")

    monkeypatch.setattr(run_drawing_diff.mp, "get_context", fail_if_started)
    assert DrawingDiffExecutor().execute([]) == []


def test_start_failure_closes_both_pipe_ends(monkeypatch, tmp_path):
    class End:
        closed = False

        def close(self):
            self.closed = True

    class Process:
        pid = None
        closed = False

        def start(self):
            raise OSError("start failed")

        def close(self):
            self.closed = True

    class Context:
        def __init__(self):
            self.receiver = End()
            self.sender = End()
            self.process = Process()

        def Pipe(self, *, duplex):
            assert duplex is False
            return self.receiver, self.sender

        def Process(self, **_kwargs):
            return self.process

    context = Context()
    monkeypatch.setattr(run_drawing_diff.mp, "get_context", lambda _method: context)
    with pytest.raises(OSError, match="start failed"):
        DrawingDiffExecutor().execute(_pairs(tmp_path))
    assert context.receiver.closed
    assert context.sender.closed
    assert context.process.closed


def test_results_keep_input_order_when_worker_sends_out_of_order(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "run_worker", _send_out_of_order)
    reports = DrawingDiffExecutor(
        timeout_seconds=20, alignment_options={"scale_tolerance": 0.02}
    ).execute(_pairs(tmp_path), drawing_scales=[0.25, None])
    assert [report.stats["index"] for report in reports] == [0, 1]
    assert [report.stats["scale"] for report in reports] == [0.25, None]
    assert all(report.stats["tolerance"] == 0.02 for report in reports)
    assert [
        (report.drawing_path, report.projection_path) for report in reports
    ] == _pairs(tmp_path)


def test_timeout_keeps_completed_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "run_worker", _send_one_then_hang)
    reports = DrawingDiffExecutor(timeout_seconds=4).execute(_pairs(tmp_path))
    assert reports[0].stats == {"completed": True}
    assert reports[1].error == "Drawing-diff timed out after 4s"
    assert (reports[1].drawing_path, reports[1].projection_path) == _pairs(tmp_path)[1]


@pytest.mark.parametrize(
    ("worker_target", "message"),
    [
        (_exit_early, "before all results arrived"),
        (_exit_after_sending, "exited with code 3"),
        (_close_early, "before all results arrived"),
        (_send_bad_index, "Invalid drawing-diff worker result"),
    ],
)
def test_broken_worker_is_not_reported_as_an_image_error(
    tmp_path, monkeypatch, worker_target, message
):
    monkeypatch.setattr(worker, "run_worker", worker_target)
    with pytest.raises(RuntimeError, match=message):
        DrawingDiffExecutor(timeout_seconds=20).execute(_pairs(tmp_path))


def test_one_bad_image_does_not_stop_other_pairs(tmp_path):
    blank = tmp_path / "blank.png"
    Image.new("RGB", (16, 16), "white").save(blank)
    reports = DrawingDiffExecutor(timeout_seconds=20).execute(
        [(tmp_path / "missing.png", blank), (blank, blank)]
    )
    assert reports[0].error.startswith("FileNotFoundError:")
    assert reports[1].error is None
    assert reports[1].alignment.status == "failed"


def test_worker_saves_diff_pngs_beside_projection(tmp_path, monkeypatch):
    drawing = np.full((64, 64, 3), 255, dtype=np.uint8)
    projection = drawing.copy()
    drawing[10:55, 20] = 0
    projection[10:55, 24] = 0
    drawing_path = tmp_path / "crop.png"
    projection_path = tmp_path / "front.png"
    Image.fromarray(drawing).save(drawing_path)
    Image.fromarray(projection).save(projection_path)
    alignment = AlignmentResult(
        "directional_chamfer", "similarity", "ok", np.eye(3).tolist(), {}
    )
    monkeypatch.setattr(worker, "align", lambda *_args, **_kwargs: alignment)

    report = worker.run_align_diff_save(drawing_path, projection_path)

    assert report.error is None
    assert report.paths == {
        "overlay_path": tmp_path / "front_overlay.png",
        "unmatched_path": tmp_path / "front_unmatched.png",
    }
    with Image.open(report.paths["overlay_path"]) as image:
        assert np.all(np.asarray(image)[10:55, 24] == [0, 168, 255])
    with Image.open(report.paths["unmatched_path"]) as image:
        assert image.mode == "RGB"


@pytest.mark.parametrize(
    ("settings", "name"),
    [
        ({"backend": None}, "backend"),
        ({"backend": "invalid"}, "backend"),
        ({"model": "invalid"}, "model"),
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"timeout_seconds": -1}, "timeout_seconds"),
        ({"distance_clip_px": 0}, "distance_clip_px"),
        ({"alignment_options": {"scale_tolerance": 0}}, "scale_tolerance"),
        ({"alignment_options": {"scale_tolerance": float("nan")}}, "scale_tolerance"),
    ],
)
def test_invalid_executor_settings_fail_at_construction(settings, name):
    with pytest.raises(ValueError, match=name):
        DrawingDiffExecutor(**settings)


@pytest.mark.parametrize(
    ("backend", "options"),
    [
        ("directional_chamfer", {"scale_step": 0}),
        ("directional_chamfer", {"outline_weight": 1.1}),
        ("match_anything", {"ransac_max_iter": 0}),
    ],
)
def test_backend_options_fail_at_construction(backend, options):
    with pytest.raises(ValueError):
        DrawingDiffExecutor(backend=backend, alignment_options=options)


def test_image_max_is_allowed():
    executor = DrawingDiffExecutor(distance_clip_px=None)
    assert executor.distance_clip_px is None


@pytest.mark.parametrize("scales", [[], [0, None], [float("nan"), None]])
def test_invalid_pair_scales_are_rejected_before_start(tmp_path, scales):
    with pytest.raises(ValueError, match="drawing_scales"):
        DrawingDiffExecutor().execute(_pairs(tmp_path), drawing_scales=scales)


@pytest.mark.parametrize("tolerance", [None, 0.02])
def test_worker_uses_dxf_raster_bounds_for_calibrated_scale(
    tmp_path, monkeypatch, tolerance
):
    import ezdxf

    drawing, projection = tmp_path / "drawing.png", tmp_path / "right.png"
    Image.new("RGB", (500, 800), "white").save(drawing)
    Image.new("RGB", (480, 640), "white").save(projection)
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline([(0, 0), (30, 0), (30, 40), (0, 40)], close=True)
    doc.saveas(projection.with_suffix(".dxf"))
    captured = []

    def align(*_args, **kwargs):
        captured.append(kwargs["options"])
        return AlignmentResult("directional_chamfer", "similarity", "failed", None, {})

    monkeypatch.setattr(worker, "align", align)
    backend_options = {"rotation_degrees": 1, "relative_scale": (0.45, 1.1)}
    options = backend_options.copy()
    if tolerance is not None:
        options["scale_tolerance"] = tolerance
    original_options = options.copy()
    report = worker.run_align_diff_save(
        drawing, projection, alignment_options=options, drawing_mm_per_pixel=0.125
    )
    # DXF margin = 30*.04=1.2 mm. Average the two raster axes' rounding error.
    source_scale = (32.4 / 480 + 42.4 / 640) / 2
    expected = source_scale / 0.125 * 640 / 800
    expected_tolerance = 0.01 if tolerance is None else tolerance
    np.testing.assert_allclose(
        captured[-1]["relative_scale"],
        [expected * (1 - expected_tolerance), expected * (1 + expected_tolerance)],
    )
    assert "scale_tolerance" not in captured[-1]
    assert options == original_options  # pair-specific, no shared mutation
    assert report.alignment.diagnostics["scale_calibration"] == {
        "status": "applied",
        "expected_pixel_scale": source_scale / 0.125,
    }

    report = worker.run_align_diff_save(
        drawing,
        projection,
        model="affine",
        alignment_options=options,
        drawing_mm_per_pixel=0.125,
    )
    assert captured[-1] == backend_options
    assert report.alignment.diagnostics["scale_calibration"]["status"] == "unavailable"

    projection.with_suffix(".dxf").unlink()
    report = worker.run_align_diff_save(
        drawing, projection, alignment_options=options, drawing_mm_per_pixel=0.125
    )
    assert captured[-1] == backend_options
    assert (
        report.alignment.diagnostics["scale_calibration"]["reason"]
        == "projection DXF unavailable"
    )
    report = worker.run_align_diff_save(drawing, projection, alignment_options=options)
    assert captured[-1] == backend_options
    assert (
        report.alignment.diagnostics["scale_calibration"]["reason"]
        == "input scale unavailable"
    )
