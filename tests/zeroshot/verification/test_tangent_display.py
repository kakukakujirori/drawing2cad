from pathlib import Path

import cv2
import ezdxf
import numpy as np
import pytest
from PIL import Image

from zeroshot.pipeline.verification import run_drawing_diff
from zeroshot.pipeline.verification.drawing_diff import worker
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult, read_rgb
from zeroshot.pipeline.verification.drawing_diff.image_ops import disk
from zeroshot.pipeline.verification.drawing_diff.tangent_display import (
    MIN_DRAWN_FRACTION,
    MIN_INFORMATIVE_PX,
    evidence_omits_tangents,
    measure_tangent_evidence,
)
from zeroshot.pipeline.verification.drawing_diff.unmatched import NEUTRAL_GRAY
from zeroshot.pipeline.verification.render.export_dxf import export_to_png

OVERLAY_BLUE = (0, 0, 128)


def _alignment(status="ok"):
    matrix = None if status == "failed" else np.eye(3).tolist()
    return AlignmentResult("directional_chamfer", "similarity", status, matrix, {})


def _line(*segments):
    mask = np.zeros((220, 220), np.uint8)
    for start, end in segments:
        cv2.line(mask, start, end, 1)
    return mask.astype(bool)


TANGENT = ((30, 100), (190, 100))
ELSEWHERE = ((30, 40), (190, 40))


def test_a_tangent_on_an_unexplained_input_line_counts_as_drawn():
    evidence = measure_tangent_evidence(
        _line(TANGENT), _line(TANGENT, ELSEWHERE), _line(ELSEWHERE)
    )
    assert evidence["drawn_px"] == evidence["informative_px"] == evidence["total_px"]
    assert evidence["total_px"] > 0


def test_a_tangent_with_no_input_line_is_informative_but_not_drawn():
    evidence = measure_tangent_evidence(
        _line(TANGENT), _line(ELSEWHERE), _line(ELSEWHERE)
    )
    assert evidence["drawn_px"] == 0
    assert evidence["informative_px"] == evidence["total_px"] > 0


def test_a_tangent_under_a_required_line_says_nothing_about_the_sheet():
    evidence = measure_tangent_evidence(_line(TANGENT), _line(TANGENT), _line(TANGENT))
    assert evidence["drawn_px"] == evidence["informative_px"] == 0


@pytest.mark.parametrize(
    "views, omits",
    [
        ([], False),
        ([(0, 0, 0)], False),
        ([(1000, MIN_INFORMATIVE_PX, 0)], True),
        ([(1000, MIN_INFORMATIVE_PX - 1, 0)], False),
        ([(1000, 1000, int(1000 * MIN_DRAWN_FRACTION) - 1)], True),
        ([(1000, 1000, int(1000 * MIN_DRAWN_FRACTION))], False),
        # Neither view has enough on its own: the decision pools the sheet.
        ([(300, 300, 0), (300, 300, 0)], True),
        # A view that draws them outweighs another whose tangents have moved.
        ([(300, 300, 300), (300, 300, 0)], False),
    ],
)
def test_the_decision_pools_every_view_of_the_sheet(views, omits):
    evidence = [
        {"total_px": total, "informative_px": informative, "drawn_px": drawn}
        for total, informative, drawn in views
    ]
    assert evidence_omits_tangents(evidence) is omits


class _Connection:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, item):
        self.sent.append(item)

    def close(self):
        self.closed = True


def _sheet(tmp_path, input_tangents, cad_tangents=((3.5, 1, 9), (13.5, 1, 9))):
    """An input drawing, its required-only render, and a build's full projection."""
    tmp_path.mkdir(exist_ok=True)
    doc = ezdxf.new()
    modelspace = doc.modelspace()
    modelspace.add_lwpolyline([(0, 0), (17, 0), (17, 10), (0, 10)], close=True)
    modelspace.add_line((0, 5), (17, 5))
    doc.saveas(tmp_path / "outline.dxf")
    outline = read_rgb(export_to_png(tmp_path / "outline.dxf"))
    added = [
        modelspace.add_line((x, start), (x, end)) for x, start, end in input_tangents
    ]
    doc.saveas(tmp_path / "drawing.dxf")
    drawing = read_rgb(export_to_png(tmp_path / "drawing.dxf"))
    for line in added:
        modelspace.delete_entity(line)
    doc.layers.add("TANGENT", color=8)
    for x, start, end in cad_tangents:
        modelspace.add_line((x, start), (x, end), dxfattribs={"layer": "TANGENT"})
    doc.saveas(tmp_path / "front.dxf")
    path = export_to_png(tmp_path / "front.dxf")
    input_path = tmp_path / "input.png"
    Image.fromarray(drawing).save(input_path)
    return input_path, outline, read_rgb(path), path


def _patch(monkeypatch, *alignments):
    """Stub OCR and alignment; return a list that records each alignment made."""
    monkeypatch.setattr(worker, "read_text_lines", lambda _drawing: [])
    made = []

    def align(*_args, **_kwargs):
        result = alignments[min(len(made), len(alignments) - 1)]
        made.append(result)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(worker, "align", align)
    return made


def _run(input_path, path, projection, monkeypatch, *, hide, status="ok"):
    """Run one entry point on a fresh projection; return its report and images."""
    Image.fromarray(projection).save(path)
    made = _patch(monkeypatch, _alignment(status))
    connection = _Connection()
    worker.run_worker([(input_path, path)], connection, auto_hide_tangents=hide)
    assert connection.closed
    assert [idx for idx, _ in connection.sent] == [0]
    report = connection.sent[0][1]
    assert report.error is None
    images = {
        "projection": read_rgb(path),
        "overlay": read_rgb(report.paths["overlay_path"]),
        "unmatched": read_rgb(report.paths["unmatched_path"]),
    }
    return report, images, len(made)


def _blank_tangent_pixels(outline, projection):
    """Tangent pixels of the projection away from every required line."""
    required = cv2.dilate((outline[..., 0] < 255).astype(np.uint8), disk(3)) > 0
    return np.any(projection != outline, axis=-1) & ~required


def test_a_sheet_that_omits_tangents_loses_them_from_every_image(tmp_path, monkeypatch):
    input_path, outline, projection, path = _sheet(tmp_path, [])
    shown, shown_images, _ = _run(input_path, path, projection, monkeypatch, hide=False)
    hidden, images, aligned = _run(input_path, path, projection, monkeypatch, hide=True)

    np.testing.assert_array_equal(images["projection"], outline)
    blank = _blank_tangent_pixels(outline, projection)
    assert blank.any()
    assert not (images["overlay"][blank] == OVERLAY_BLUE).all(axis=-1).any()
    assert not (images["unmatched"][blank] == NEUTRAL_GRAY).all(axis=-1).any()
    # The feature changes the pictures only: one alignment, the same measurements.
    assert aligned == 1
    assert hidden.stats == shown.stats
    assert (shown_images["unmatched"][blank] == NEUTRAL_GRAY).all(axis=-1).any()


def test_a_sheet_that_draws_tangents_keeps_every_one_however_far_it_moved(
    tmp_path, monkeypatch
):
    # The input draws one tangent in place and the other 24 px from the CAD edge.
    input_path, _outline, projection, path = _sheet(
        tmp_path, [(3.5, 1, 9), (14.0, 1, 9)]
    )
    shown, shown_images, _ = _run(input_path, path, projection, monkeypatch, hide=False)
    report, images, _ = _run(input_path, path, projection, monkeypatch, hide=True)

    for name, image in images.items():
        np.testing.assert_array_equal(image, shown_images[name])
    assert report.stats == shown.stats
    np.testing.assert_array_equal(images["projection"], projection)


def test_too_little_evidence_leaves_tangents_drawn(tmp_path, monkeypatch):
    input_path, _outline, projection, path = _sheet(
        tmp_path, [], cad_tangents=[(3.5, 1, 3)]
    )
    _, images, _ = _run(input_path, path, projection, monkeypatch, hide=True)

    np.testing.assert_array_equal(images["projection"], projection)


def test_a_view_without_a_sound_alignment_gives_no_evidence(tmp_path, monkeypatch):
    input_path, _outline, projection, path = _sheet(tmp_path, [])
    _, images, _ = _run(
        input_path, path, projection, monkeypatch, hide=True, status="uncertain"
    )

    np.testing.assert_array_equal(images["projection"], projection)


def test_a_failed_view_of_a_sheet_that_omits_tangents_shows_none_either(
    tmp_path, monkeypatch
):
    good = _sheet(tmp_path / "good", [])
    failed = _sheet(tmp_path / "failed", [])
    _patch(monkeypatch, _alignment(), _alignment("failed"))
    connection = _Connection()

    worker.run_worker(
        [(good[0], good[3]), (failed[0], failed[3])],
        connection,
        auto_hide_tangents=True,
    )

    assert [idx for idx, _ in connection.sent] == [0, 1]
    assert connection.sent[1][1].paths == {}
    np.testing.assert_array_equal(read_rgb(failed[3]), failed[1])


def test_a_view_that_raises_is_reported_and_the_rest_still_run(tmp_path, monkeypatch):
    broken = _sheet(tmp_path / "broken", [])
    good = _sheet(tmp_path / "good", [])
    _patch(monkeypatch, RuntimeError("cannot align"), _alignment())
    connection = _Connection()

    worker.run_worker(
        [(broken[0], broken[3]), (good[0], good[3])],
        connection,
        auto_hide_tangents=True,
    )

    assert connection.sent[0][1].error == "RuntimeError: cannot align"
    assert connection.sent[1][1].error is None
    assert connection.closed
    np.testing.assert_array_equal(read_rgb(good[3]), good[1])


@pytest.mark.parametrize("flag", [True, False])
def test_the_executor_hands_the_yaml_flag_to_its_worker(monkeypatch, flag):
    started = {}

    class End:
        def close(self):
            pass

    class Context:
        def Pipe(self, *, duplex):
            return End(), End()

        def Process(self, **kwargs):
            started.update(kwargs)
            raise OSError("stop before starting")

    monkeypatch.setattr(run_drawing_diff.mp, "get_context", lambda _method: Context())
    pairs = [(Path("missing_input.png"), Path("missing_front.png"))]

    with pytest.raises(OSError, match="stop before starting"):
        run_drawing_diff.DrawingDiffExecutor(auto_hide_tangents=flag).execute(pairs)

    assert started["kwargs"]["auto_hide_tangents"] is flag
    assert started["target"].__name__ == "run_worker"
