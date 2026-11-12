from dataclasses import replace

import cadquery as cq
import pytest

from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import (
    VerifyOutputResult,
    build_verification_feedback,
)
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
)
from zeroshot.pipeline.verification.shape_census import read_census
from zeroshot.pipeline.verification.shape_warnings import find_shape_warnings


def _thin_land():
    body = cq.Solid.makeBox(40, 30, 30)
    lower = cq.Solid.makeBox(35, 20, 10, cq.Vector(5, 5, 0))
    upper = cq.Solid.makeBox(35, 20, 9.9, cq.Vector(5, 5, 10.1))
    return body.cut(lower).cut(upper)


@pytest.mark.parametrize(
    "method", ["spline", "splineApprox", "makeSpline", "makeSplineApprox"]
)
def test_spline_warning_names_the_method_line_in_a_multiline_chain(method):
    source = (
        "import cadquery as cq\n"
        "result = (\n"
        "    cq.Workplane()\n"
        f"    .{method}(\n"
        "        points\n"
        "    )\n"
        ")\n"
    )

    warnings = find_shape_warnings(
        cq.Solid.makeBox(10, 20, 30), source, filename="profile.py"
    )

    assert len(warnings) == 1
    assert f"profile.py:4 ({method})" in warnings[0]
    assert "freeform" in warnings[0]
    assert "more points" in warnings[0]


def test_spline_warning_combines_locations_without_matching_comments_or_strings():
    shape = cq.Solid.makeBox(10, 20, 30)
    source = (
        "# cq.Edge.makeSpline(points)\n"
        'label = "wp.spline(points)"\n'
        "a = wp.spline(points)\n"
        "b = cq.Edge.makeSpline(points)\n"
    )

    warnings = find_shape_warnings(shape, source)

    assert len(warnings) == 1
    assert "model.py:3 (spline)" in warnings[0]
    assert "model.py:4 (makeSpline)" in warnings[0]
    assert "model.py:1" not in warnings[0] and "model.py:2" not in warnings[0]
    assert (
        find_shape_warnings(
            shape,
            "result = cq.Workplane().box(10, 20, 30)\n"
            '# wp.spline(points)\nlabel = "makeSpline"\n',
        )
        == ()
    )


def test_source_and_solid_checks_share_the_warning_result(tmp_path):
    path = tmp_path / "output.step"
    cq.exporters.export(_thin_land(), str(path))
    geometry_only = read_census(path)

    reviewed = read_census(
        path, source="result = wp.spline(points)", filename="part.py"
    )

    assert geometry_only is not None and len(geometry_only.warnings) == 1
    assert reviewed is not None
    assert reviewed.warnings[:-1] == geometry_only.warnings
    assert "part.py:1 (spline)" in reviewed.warnings[-1]
    assert replace(reviewed, warnings=()) == replace(geometry_only, warnings=())


@pytest.mark.parametrize("scale", [0.01, 1, 100])
def test_a_local_land_is_detected_without_changing_the_shape(scale):
    shape = _thin_land().scale(scale)
    before = (shape.Volume(), len(shape.Faces()), len(shape.Edges()))

    warnings = find_shape_warnings(shape)

    assert len(warnings) == 1
    assert warnings[0].startswith("Possible local thin lands (1 region):")
    assert f"thickness {0.1 * scale:.6g} mm" in warnings[0]
    assert f"z={10 * scale:.6g}..{10.1 * scale:.6g}" in warnings[0]
    assert (shape.Volume(), len(shape.Faces()), len(shape.Edges())) == before


def test_the_thin_land_check_is_independent_of_the_model_frame():
    shape = _thin_land().rotate((0, 0, 0), (1, 2, 3), 37).translate((100, -50, 200))

    warnings = find_shape_warnings(shape)

    assert len(warnings) == 1
    assert warnings[0].startswith("Possible local thin lands")
    assert "thickness 0.1 mm" in warnings[0]


@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("scale", [0.01, 1, 100])
def test_a_shallow_recess_is_located(circular, scale):
    body = cq.Solid.makeBox(40, 30, 30)
    tool = (
        cq.Solid.makeCylinder(3, 0.1, cq.Vector(20, 15, 29.9))
        if circular
        else cq.Solid.makeBox(0.1, 30, 10, cq.Vector(39.9, 0, 20))
    )

    warnings = find_shape_warnings(body.cut(tool).scale(scale))

    assert len(warnings) == 1
    assert warnings[0].startswith("Possible shallow steps (1 region)")
    assert f"depth {0.1 * scale:.6g} mm" in warnings[0]
    span = (
        f"z={29.9 * scale:.6g}..{30 * scale:.6g}"
        if circular
        else f"x={39.9 * scale:.6g}..{40 * scale:.6g}"
    )
    assert span in warnings[0]


@pytest.mark.parametrize("rotated", [False, True])
def test_a_curved_end_face_is_not_mistaken_for_a_shallow_step(rotated):
    shape = (
        cq.Workplane("XZ")
        .moveTo(0, 10)
        .lineTo(10, 10)
        .lineTo(10, 9.9)
        .lineTo(20, 9.9)
        .threePointArc((10, -10), (0, 10))
        .close()
        .extrude(30)
        .val()
    )
    if rotated:
        shape = shape.rotate((0, 0, 0), (1, 2, 3), 37).translate((100, -50, 200))

    assert shape.isValid()
    warnings = find_shape_warnings(shape)

    # The actual 0.1 mm riser qualifies; the two tall curved end faces do not.
    assert len(warnings) == 1
    assert warnings[0].startswith("Possible shallow steps (1 region): depth 0.1 mm")


@pytest.mark.parametrize(
    "shape",
    [
        cq.Solid.makeBox(40, 30, 0.1),  # Uniform thin material is not a local anomaly.
        cq.Solid.makeBox(40, 30, 30).cut(
            cq.Solid.makeBox(35, 20, 0.1, cq.Vector(5, 5, 10))
        ),  # Close opposing faces across a void are not a thin wall.
        cq.Solid.makeBox(40, 30, 30).cut(
            cq.Solid.makeBox(3, 30, 10, cq.Vector(37, 0, 20))
        ),
        cq.Solid.makeCylinder(5, 20),  # A cylinder's parameter seam is not a split.
        cq.Workplane().box(20, 30, 40).edges("|Z").fillet(2).val(),
    ],
)
def test_regular_solids_and_real_thin_sheets_need_no_warning(shape):
    assert find_shape_warnings(shape) == ()


@pytest.mark.parametrize("cylindrical", [False, True])
def test_only_shared_same_surface_partitions_are_flagged(cylindrical):
    shape = (
        cq.Workplane().circle(5).extrude(10, both=True).val()
        if cylindrical
        else cq.Solid.makeBox(10, 10, 5).fuse(
            cq.Solid.makeBox(10, 10, 5, cq.Vector(0, 0, 5))
        )
    )

    warnings = find_shape_warnings(shape)

    assert len(warnings) == 1
    assert warnings[0].startswith("Adjacent faces on the same analytic surface")
    assert ("cylinder partition" if cylindrical else "plane partition") in warnings[0]
    assert ("z=0..0" if cylindrical else "z=5..5") in warnings[0]


def test_meridian_halves_of_a_step_cylinder_are_not_construction_warnings():
    full = cq.Solid.makeCylinder(5, 20)
    half = cq.Solid.makeCylinder(5, 20, angleDegrees=180)
    other_half = half.rotate((0, 0, 0), (0, 0, 1), 180)
    caps = [face for face in full.Faces() if face.geomType() == "PLANE"]
    sides = [
        face
        for part in (half, other_half)
        for face in part.Faces()
        if face.geomType() == "CYLINDER"
    ]
    shape = cq.Solid.makeSolid(cq.Shell.makeShell([*caps, *sides]))

    assert shape.isValid() and len(shape.Faces()) == 4
    assert find_shape_warnings(shape) == ()


def test_warning_survives_step_and_is_only_shown_when_present(tmp_path):
    path = tmp_path / "output.step"
    cq.exporters.export(_thin_land(), str(path))
    census = read_census(path)
    assert census is not None and census.warnings
    report = VerifyOutputResult(
        exec_report=CadQueryExecutionReport(
            status=ExecutionStatus.VERIFIED, step_path=path, census=census
        )
    )

    def feedback(candidate):
        blocks = build_verification_feedback(
            candidate,
            SandboxWorkdir(host_bind_dir=tmp_path),
            ArtifactPresenter(input="image"),
            {},
            own_program="model.py",
        )
        return "\n".join(b.get("text", "") for b in blocks)

    text = feedback(report)
    assert "[Geometry warnings" in text
    assert "thickness 0.1 mm" in text
    assert "not automatic removal" in text
    assert report.exec_report.status is ExecutionStatus.VERIFIED

    without = replace(
        report,
        exec_report=replace(report.exec_report, census=replace(census, warnings=())),
    )
    assert "[Geometry warnings" not in feedback(without)


def test_a_review_failure_does_not_discard_the_census(tmp_path, monkeypatch):
    from zeroshot.pipeline.verification import shape_warnings

    path = tmp_path / "output.step"
    cq.exporters.export(cq.Solid.makeBox(10, 20, 30), str(path))

    def broken_review(shape, source=None, *, filename="model.py"):
        raise RuntimeError("Review failed")

    monkeypatch.setattr(shape_warnings, "find_shape_warnings", broken_review)
    census = read_census(path)

    assert census is not None and census.volume == pytest.approx(6000)
    assert census.warnings is None
