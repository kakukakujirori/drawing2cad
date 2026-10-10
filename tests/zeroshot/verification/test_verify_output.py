import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Literal

import ezdxf
import pytest
from PIL import Image
from pydantic import TypeAdapter

from tests.zeroshot.contracts import UNTURNED, interpretation
from tests.zeroshot.contracts import view as drawing_view
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.stages.coding.verify import (
    FEEDBACK_PICTORIAL,
    RESULT_NAME,
    OutputVerifier,
    VerifyOutputResult,
)
from zeroshot.pipeline.stages.interpretation.contracts import Axis, Region, View
from zeroshot.pipeline.stages.tickets.contracts import TicketAnswers
from zeroshot.pipeline.verification.attempts import AttemptStore
from zeroshot.pipeline.verification.drawing_diff.align import AlignmentResult
from zeroshot.pipeline.verification.render.constants import (
    ProjectionPaths,
    Render3dPaths,
)
from zeroshot.pipeline.verification.render.orthographic import STANDARD_VIEW_FRAMES
from zeroshot.pipeline.verification.run_cadquery import (
    CadQueryExecutionReport,
    ExecutionStatus,
)
from zeroshot.pipeline.verification.run_drawing_diff import DrawingDiffReport
from zeroshot.pipeline.verification.run_render import (
    RenderReport,
    RenderRequest,
    RenderStatus,
)
from zeroshot.pipeline.verification.shape_census import read_census

RENDER3D_STYLES = (
    "hlg_perspective",
    "transparent_shaded_edges_perspective",
    "hlg_translucent_faces_perspective",
)

VIEWS = ("front", "top", "right")
VIEW_FRAMES = {View(view): STANDARD_VIEW_FRAMES[View(view)] for view in VIEWS}

VALID_SOURCE = """\
import cadquery as cq

result = cq.Workplane("XY").box(10, 20, 30)
"""


class StubCadQueryExecutor:
    def __init__(self, report: CadQueryExecutionReport) -> None:
        self.report = report
        self.calls: list[tuple[Path, Path | None]] = []

    def execute(
        self, model_path: Path, output_step_path: Path | None = None
    ) -> CadQueryExecutionReport:
        self.calls.append((model_path, output_step_path))
        report = self.report
        # A verified run leaves the STEP behind, which is what gets rendered.
        if (
            output_step_path is not None
            and self.report.status is ExecutionStatus.VERIFIED
        ):
            output_step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n")
            report = replace(report, step_path=output_step_path)
        return report


def _execution_report(
    *,
    source: str | None = VALID_SOURCE,
    status: ExecutionStatus = ExecutionStatus.VERIFIED,
    returncode: int | None = 0,
    stdout: str = "construction log",
    stderr: str = "",
    executor_error: str | None = None,
) -> CadQueryExecutionReport:
    return CadQueryExecutionReport(
        source=source,
        status=status,
        executor_error=executor_error,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _create_verifier(
    executor: StubCadQueryExecutor,
    workdir: SandboxWorkdir,
    *,
    renderer: object | None = None,  # defaults to a StubRenderer
    output_renders: Literal["path", "image"] = "path",
    views: Mapping[View, tuple[str, str]] = VIEW_FRAMES,
    source_filename: str = "model.py",
    output_dirname: PurePosixPath = PurePosixPath("attempts"),
    attempt_store: AttemptStore | None = None,
    diff_drawer: object | None = None,
    projection_view_mode: Literal["interpreted", "standard"] = "interpreted",
    unmatched: Literal["path", "image"] = "image",
    overlay: Literal["none", "path", "image"] = "path",
) -> OutputVerifier:
    verifier = OutputVerifier(
        executor,  # type: ignore[arg-type]
        workdir,
        renderer=renderer or StubRenderer(),  # type: ignore[arg-type]
        artifact_presenter=ArtifactPresenter(
            input="path",
            output_renders=output_renders,
            unmatched=unmatched,
            overlay=overlay,
        ),
        attempt_store=attempt_store
        or AttemptStore(
            workdir,
            round_source=lambda: 0,
            root_dirname=output_dirname,
        ),
        source_filename=source_filename,
        diff_drawer=diff_drawer,  # type: ignore[arg-type]
        projection_view_mode=projection_view_mode,
    )
    if projection_view_mode == "interpreted":
        verifier.interpretation = interpretation(
            views=[
                drawing_view(role.value, u_axis=axes[0], v_axis=axes[1])
                for role, axes in views.items()
            ]
        )
    return verifier


def _coding_attempt(
    workdir: Path,
    attempt_id: str = "000",
    round_number: int = 0,
) -> Path:
    return workdir / "attempts" / f"round_{round_number:03d}" / "coding" / attempt_id


@pytest.mark.parametrize(
    ("source", "status", "solids"),
    [
        (VALID_SOURCE, ExecutionStatus.VERIFIED, 1),
        (
            (
                "import cadquery as cq\n"
                "result = cq.Compound.makeCompound([\n"
                "    cq.Workplane('XY').box(10, 20, 30).val(),\n"
                "    cq.Workplane('XY').box(1, 1, 1).translate((15, 0, 0)).val(),\n"
                "])\n"
            ),
            ExecutionStatus.FAILED,
            2,
        ),
    ],
)
def test_real_cadquery_render_and_drawing_diff_reach_feedback(
    tmp_path: Path, source: str, status: ExecutionStatus, solids: int
) -> None:
    import sys

    from PIL import ImageDraw

    from zeroshot.pipeline.sandbox import SandboxRunner
    from zeroshot.pipeline.verification import (
        CadQueryExecutor,
        DrawingDiffExecutor,
        StepRenderer,
    )

    (tmp_path / "model.py").write_text(source)
    drawing_path = tmp_path / "front.png"
    image = Image.new("RGB", (140, 340), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 120, 320), outline="black", width=2)
    image.save(drawing_path)
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        verifier = OutputVerifier(
            executor=CadQueryExecutor(
                sandbox_runner=SandboxRunner(
                    python_executable=Path(sys.executable), default_timeout_s=30
                )
            ),
            workdir=workdir,
            renderer=StepRenderer(max_workers=1),
            diff_drawer=DrawingDiffExecutor(),
            artifact_presenter=ArtifactPresenter(input="path", unmatched="image"),
            attempt_store=AttemptStore(workdir, round_source=lambda: 0),
        )
        verifier.interpretation = interpretation(
            views=[
                drawing_view(
                    file="/work/front.png",
                    image_size=image.size,
                    region=Region(view="view_front", box_px=(0, 0, *image.size)),
                )
            ],
        )

        blocks = verifier.feedback()
        report = verifier.verify()

    assert report.exec_report.status is status
    assert report.exec_report.census.solids == solids
    assert report.exec_report.step_path.is_file()
    assert verifier.confirmed is (status is ExecutionStatus.VERIFIED)
    assert verifier.accepted_source == (
        source if status is ExecutionStatus.VERIFIED else None
    )
    assert report.render_report[RESULT_NAME].projection_paths.front.is_file()
    diff = report.drawing_diff_report["view_front"]
    assert diff.error is None
    assert diff.alignment.H_drawing_to_projection is not None
    assert diff.drawing_path == drawing_path
    assert set(diff.paths) == {"overlay_path", "unmatched_path"}
    adapter = TypeAdapter(VerifyOutputResult)
    restored = adapter.validate_json(adapter.dump_json(report))
    assert restored.exec_report == report.exec_report
    assert restored.render_report == report.render_report
    assert restored.drawing_diff_report["view_front"].paths == diff.paths
    # Untyped diagnostic tuples become JSON arrays; their values must survive.
    assert adapter.dump_json(restored) == adapter.dump_json(report)
    # The unmatched image is drawn on the input, the overlay on the projection.
    frames = {
        "overlay_path": diff.projection_path,
        "unmatched_path": diff.drawing_path,
    }
    for key, path in diff.paths.items():
        assert path.parent == diff.projection_path.parent
        with Image.open(path) as saved, Image.open(frames[key]) as frame:
            assert saved.size == frame.size
    text = "\n".join(block["text"] for block in blocks if block["type"] == "text")
    if status is ExecutionStatus.FAILED:
        assert "Expected exactly one solid, found 2" in text
        assert "[Diagnostic result]" in text
    assert "[Drawing comparison]" in text
    assert "/work/attempts/round_000/coding/000/projection/front_overlay.png" in text
    assert "/work/attempts/round_000/coding/000/projection/front_unmatched.png" in text
    # The unmatched image is attached right after the line that names it.
    label = next(
        index
        for index, block in enumerate(blocks)
        if block["type"] == "text" and block["text"].startswith("view_front unmatched:")
    )
    assert blocks[label + 1]["type"] == "image"


type Frames = Mapping[View, tuple[Axis, Axis]]


class StubRenderer:
    """A ``StepRenderer`` that writes placeholder artifacts instead of rendering.

    ``skip_styles`` drops perspective styles the way a partial render does, so
    the manifest can be checked for reporting only what actually exists.
    ``corrupt_views`` writes an unreadable file where a projection should be.
    """

    def __init__(
        self,
        *,
        skip_styles: tuple[str, ...] = (),
        corrupt_views: tuple[str, ...] = (),
    ) -> None:
        self.skip_styles = skip_styles
        self.corrupt_views = corrupt_views
        self.calls: list[tuple[Path, ProjectionPaths, Render3dPaths, Frames]] = []

    def render_many(self, requests: Sequence[RenderRequest]) -> list[RenderReport]:
        return [
            self.render(
                request.step_path,
                request.projection_paths,
                request.render3d_paths,
                request.frames,
            )
            for request in requests
        ]

    def render(
        self,
        input_step_path: Path,
        output_projection_paths: ProjectionPaths,
        output_render3d_paths: Render3dPaths,
        frames: Frames,
    ) -> RenderReport:
        self.calls.append(
            (input_step_path, output_projection_paths, output_render3d_paths, frames)
        )
        for view, path in output_projection_paths.as_mapping().items():
            if view in self.corrupt_views:
                path.write_text("not a drawing", encoding="utf-8")
                continue
            doc = ezdxf.new()
            # Projections keep the model frame, which may extend below zero.
            doc.modelspace().add_lwpolyline(
                [(-5, -5), (10, -5), (10, 10), (-5, 10)], close=True
            )
            doc.saveas(path)
            Image.new("RGB", (20, 20), "white").save(path.with_suffix(".png"))

        errors: dict[str, str] = {}
        for style in self.skip_styles:
            output_render3d_paths = replace(output_render3d_paths, **{style: None})
            errors[style] = f"RuntimeError: {style} failed"
        for style, path in output_render3d_paths.as_mapping().items():
            assert style not in self.skip_styles
            Image.new("RGB", (20, 20), "white").save(path)

        return RenderReport(
            status=RenderStatus.OK if not errors else RenderStatus.PARTIAL,
            projection_paths=output_projection_paths,
            render3d_paths=output_render3d_paths,
            render3d_errors=errors,
        )


def _text(result: object) -> str:
    """Concatenate the text the model would read out of the tool result."""
    assert isinstance(result, list)
    return "\n".join(b["text"] for b in result if b["type"] == "text")


def _report_json(result: object) -> dict:
    """The verification report the tool result leads with."""
    text = _text(result)
    start = text.index("{")
    return json.loads(text[start : text.rindex("}") + 1])


def test_construction_prepares_an_output_directory_the_sandbox_cannot_write(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    verifier = _create_verifier(executor, workdir, source_filename="candidate.py")

    assert verifier.source_path == tmp_path / "candidate.py"
    assert (tmp_path / "attempts").is_dir()
    assert workdir.read_only_subdirs == [PurePosixPath("attempts")]


@pytest.mark.parametrize(
    "source_filename",
    ["", ".", "..", "../model.py", "nested/model.py", "/work/model.py", "model.txt"],
)
def test_rejects_a_source_filename_outside_the_workdir_root_or_not_python(
    tmp_path: Path,
    source_filename: str,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(
        ValueError, match="source_filename must be a Python file basename"
    ):
        _create_verifier(executor, workdir, source_filename=source_filename)


def test_delegates_paths_and_returns_json_safe_mapping(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            returncode=0,
            stdout="construction log",
            stderr="construction warning",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert executor.calls == [
        (
            tmp_path / "model.py",
            _coding_attempt(tmp_path) / "output.step",
        )
    ]
    assert _report_json(result) == {
        "verification_id": "000",
        "status": "VERIFIED",
        "returncode": 0,
        "stdout": "construction log",
        "stderr": "construction warning",
        "executor_error": None,
        "shape": "",
    }
    report = _report_json(result)
    assert isinstance(report["returncode"], int)
    assert "source" not in report
    assert (_coding_attempt(tmp_path) / "model.py").read_text(
        encoding="utf-8"
    ) == VALID_SOURCE


def test_preserves_failed_attempt_and_execution_report(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            status=ExecutionStatus.FAILED,
            returncode=1,
            stdout="partial output",
            stderr="execution failed",
            executor_error="output.step was not generated",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result) == {
        "verification_id": "000",
        "status": "FAILED",
        "returncode": 1,
        "stdout": "partial output",
        "stderr": "execution failed",
        "executor_error": "output.step was not generated",
        "shape": "",
    }
    attempt_dir = _coding_attempt(tmp_path)
    assert (attempt_dir / "model.py").read_text(encoding="utf-8") == VALID_SOURCE
    assert not (attempt_dir / "output.step").exists()


def test_assigns_incrementing_verification_ids(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    first = verifier.feedback()
    (tmp_path / "model.py").write_text(
        VALID_SOURCE.replace("10, 20, 30", "10, 20, 40"), encoding="utf-8"
    )
    second = verifier.feedback()

    assert _report_json(first)["verification_id"] == "000"
    assert _report_json(second)["verification_id"] == "001"
    assert _coding_attempt(tmp_path, "000").is_dir()
    assert _coding_attempt(tmp_path, "001").is_dir()


def test_a_new_round_restarts_coding_attempt_ids_without_overwriting_history(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    current_round = [0]
    store = AttemptStore(workdir, round_source=lambda: current_round[0])
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir, attempt_store=store)
    first = verifier.verify()

    current_round[0] = 1
    verifier.reset()
    second = verifier.verify()

    assert first.verification_id == second.verification_id == "000"
    assert _coding_attempt(tmp_path, round_number=0).is_dir()
    assert _coding_attempt(tmp_path, round_number=1).is_dir()


def test_an_unchanged_program_is_not_built_twice(tmp_path: Path) -> None:
    """Two verifications of one program: one build, one attempt directory."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    first = verifier.verify()
    second = verifier.verify()

    assert len(executor.calls) == 1
    assert first == second
    assert [path.name for path in (tmp_path / "attempts").iterdir()] == ["round_000"]
    assert [path.name for path in _coding_attempt(tmp_path).parent.iterdir()] == ["000"]


class StubDiffDrawer:
    def __init__(self, *, error=None, fatal=False, chamfers=()):
        self.error = error
        self.fatal = fatal
        self.chamfers = list(chamfers)  # one per call; None measures nothing
        self.calls = []
        self.drawing_scales = []

    def execute(self, pairs, *, drawing_scales=None):
        self.calls.append(list(pairs))
        self.drawing_scales.append(drawing_scales)
        if self.fatal:
            raise RuntimeError("comparison worker failed")
        chamfer = self.chamfers.pop(0) if self.chamfers else None
        if not self.error:  # saved as the real worker saves it
            for _, projection in pairs:
                unmatched = projection.with_name(f"{projection.stem}_unmatched.png")
                Image.new("RGB", (4, 4), "white").save(unmatched)
                Image.new("RGB", (4, 4), "white").save(
                    projection.with_name(f"{projection.stem}_overlay.png")
                )
        return [
            DrawingDiffReport(
                drawing_path=drawing,
                projection_path=projection,
                error=self.error,
                stats={"bounded_chamfer_drawing_px": chamfer},
                alignment=None
                if self.error
                else AlignmentResult(
                    "directional_chamfer",
                    "similarity",
                    "ok",
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    {},
                ),
                paths={}
                if self.error
                else {
                    "overlay_path": projection.with_name(
                        f"{projection.stem}_overlay.png"
                    ),
                    "unmatched_path": projection.with_name(
                        f"{projection.stem}_unmatched.png"
                    ),
                },
            )
            for drawing, projection in pairs
        ]


def _set_input_crop(verifier, tmp_path):
    Image.new("RGB", (20, 20), "white").save(tmp_path / "front.png")
    verifier.interpretation = interpretation(
        views=[
            drawing_view(
                "full_page",
                name="view_page",
                file="/work/page.png",
                region=Region(view="view_page", box_px=(0, 0, 1000, 1000)),
            ),
            drawing_view(
                "front",
                file="/work/front.png",
                # DrawingView.file is already cropped; this box locates it on the page.
                region=Region(view="view_page", box_px=(800, 600, 820, 620)),
            ),
        ]
    )


def test_feedback_compares_scores_with_the_last_scored_build(tmp_path):
    workdir = SandboxWorkdir(tmp_path)
    drawer = StubDiffDrawer(chamfers=[8.0, None, 6.0, 7.0])
    verifier = _create_verifier(
        StubCadQueryExecutor(_execution_report()), workdir, diff_drawer=drawer
    )
    _set_input_crop(verifier, tmp_path)

    def build(number: int) -> str:
        (tmp_path / "model.py").write_text(f"{VALID_SOURCE}\n# build {number}\n")
        return _text(verifier.feedback())

    first = build(0)
    assert "line distance 8.00 px; silhouette mismatch unavailable" in first
    assert "Changes in parentheses" not in first
    unscored = build(1)
    assert "line distance unavailable" in unscored
    assert "Mean line distance over 0 views: unavailable" in unscored
    third = build(2)
    assert "line distance 6.00 px (-2.00)" in third
    assert "against verification 000" in third
    assert _text(verifier.feedback()) == third  # not against itself

    verifier.reset()
    assert "Changes in parentheses" not in build(3)


@pytest.mark.parametrize("unmatched", ["path", "image"])
def test_diff_uses_final_render_and_feedback_formats_the_stored_reports(
    tmp_path, unmatched
):
    workdir = SandboxWorkdir(tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE)
    executor = StubCadQueryExecutor(_execution_report())
    drawer = StubDiffDrawer()
    verifier = _create_verifier(
        executor,
        workdir,
        diff_drawer=drawer,
        unmatched=unmatched,
        output_renders="path",
    )
    _set_input_crop(verifier, tmp_path)
    verifier.interpretation.views[1].scale = 0.125

    report = verifier.verify()
    assert set(report.render_report) == {RESULT_NAME}
    assert isinstance(report.drawing_diff_report["view_front"], DrawingDiffReport)
    assert drawer.calls == [
        [(tmp_path / "front.png", _coding_attempt(tmp_path) / "projection/front.png")]
    ]
    assert drawer.drawing_scales == [[0.125]]
    assert not verifier.confirmed  # Only feedback confirms the inspected build.

    blocks = verifier.feedback()
    text = _text(blocks)
    headings = [
        "[Execution result]",
        "[Projected drawing]",
        "[Drawing comparison]",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert all(text.count(heading) == 1 for heading in headings)
    assert sum(block["type"] == "image" for block in blocks) == int(
        unmatched == "image"
    )
    assert verifier.confirmed
    assert "view_front" in text
    assert "projection/front_overlay.png" in text
    assert "projection/front_unmatched.png" in text
    assert "drawing_diff.json" not in text
    assert str(tmp_path) not in text
    assert len(executor.calls) == len(drawer.calls) == 1
    assert verifier.verify() is report

    # All three nested reports retain their types and values through JSON.
    adapter = TypeAdapter(VerifyOutputResult)
    restored = adapter.validate_json(adapter.dump_json(report))
    assert restored == report
    assert isinstance(
        restored.drawing_diff_report["view_front"].alignment, AlignmentResult
    )
    assert isinstance(restored.exec_report.step_path, Path)
    assert _text(verifier.feedback()).count("[Drawing comparison]") == 1

    # A new stage invocation resets cached builds, even if model.py is unchanged.
    verifier.reset()
    verifier.feedback()
    assert len(executor.calls) == len(drawer.calls) == 2


def test_unavailable_views_keep_reasons_without_entering_the_worker(tmp_path):
    (tmp_path / "model.py").write_text(VALID_SOURCE)
    drawer = StubDiffDrawer()
    verifier = _create_verifier(
        StubCadQueryExecutor(_execution_report()),
        SandboxWorkdir(tmp_path),
        renderer=StubRenderer(corrupt_views=("top",)),
        diff_drawer=drawer,
    )
    _set_input_crop(verifier, tmp_path)
    (tmp_path / "right.dxf").write_text("native drawing")
    page, front = verifier.interpretation.views
    verifier.interpretation = interpretation(
        views=[
            page,
            front,
            front.model_copy(update={"name": "view_front_detail"}),
            drawing_view("top", file="/work/front.png"),
            drawing_view("right", file="/work/right.dxf"),
            drawing_view("left", file="/work/missing.png"),
        ]
    )

    reports = verifier.verify().drawing_diff_report

    assert len(drawer.calls) == 1 and len(drawer.calls[0]) == 1
    assert "view_page" not in reports
    assert reports["view_front"].error is None
    assert "already uses this role" in reports["view_front_detail"].error
    assert "projection PNG unavailable" in reports["view_top"].error
    assert "native DXF" in reports["view_right"].error
    assert "input drawing unavailable" in reports["view_left"].error


def test_source_digest_reads_only_model_source(tmp_path):
    verifier = _create_verifier(
        StubCadQueryExecutor(_execution_report()), SandboxWorkdir(tmp_path)
    )
    assert verifier.source_digest() is None
    verifier.source_path.write_text(VALID_SOURCE)
    original = verifier.source_digest()
    _set_input_crop(verifier, tmp_path)
    assert verifier.source_digest() == original
    verifier.source_path.write_text(VALID_SOURCE + "\n# changed program\n")
    assert verifier.source_digest() != original


@pytest.mark.parametrize("enabled", [False, True])
def test_pair_failure_is_feedback_and_does_not_reject_step(tmp_path, enabled):
    (tmp_path / "model.py").write_text(VALID_SOURCE)
    drawer = StubDiffDrawer(error="alignment failed") if enabled else None
    verifier = _create_verifier(
        StubCadQueryExecutor(_execution_report()),
        SandboxWorkdir(tmp_path),
        diff_drawer=drawer,
        output_renders="path",
    )
    _set_input_crop(verifier, tmp_path)

    text = _text(verifier.feedback())
    report = verifier.verify()
    assert report.exec_report.status is ExecutionStatus.VERIFIED
    assert verifier.confirmed
    assert ("alignment failed" in text) is enabled
    assert ("[Drawing comparison]" in text) is enabled
    assert "Overlay images:" not in text
    assert "Unmatched images:" not in text
    assert "projection/front.png" in text
    assert (report.drawing_diff_report is not None) is enabled


def test_fatal_worker_failure_is_not_hidden_as_a_pair_warning(tmp_path):
    (tmp_path / "model.py").write_text(VALID_SOURCE)
    verifier = _create_verifier(
        StubCadQueryExecutor(_execution_report()),
        SandboxWorkdir(tmp_path),
        diff_drawer=StubDiffDrawer(fatal=True),
    )
    _set_input_crop(verifier, tmp_path)
    with pytest.raises(RuntimeError, match="comparison worker failed"):
        verifier.feedback()
    assert not verifier.confirmed


def test_a_program_written_after_a_failed_verification_is_built(
    tmp_path: Path,
) -> None:
    """Nothing on disk is nothing to reuse, so the program that arrives is built."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verifier = _create_verifier(executor, workdir)

    missing = verifier.verify()
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    written = verifier.verify()

    assert missing.exec_report.status is ExecutionStatus.REJECTED
    assert written.exec_report.status is ExecutionStatus.VERIFIED
    assert len(executor.calls) == 1


def test_rejects_missing_source_without_issuing_id(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result) == {
        "verification_id": None,
        "status": "REJECTED",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "executor_error": "model.py was not found",
        "shape": "",
    }
    assert executor.calls == []
    assert list((tmp_path / "attempts").iterdir()) == []


def test_rejects_source_symlink_without_issuing_id(tmp_path: Path) -> None:
    real_source = tmp_path / "real-model.py"
    real_source.write_text(VALID_SOURCE, encoding="utf-8")
    (tmp_path / "model.py").symlink_to(real_source)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result)["verification_id"] is None
    assert _report_json(result)["status"] == "REJECTED"
    assert _report_json(result)["executor_error"] == "model.py must not be a symlink"
    assert executor.calls == []
    assert list((tmp_path / "attempts").iterdir()) == []


def test_preserves_executor_rejection_without_source_snapshot(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(
            source=None,
            status=ExecutionStatus.REJECTED,
            returncode=None,
            stdout="",
            stderr="",
            executor_error="model.py must be valid UTF-8",
        )
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_bytes(b"\xff")
    verifier = _create_verifier(executor, workdir)

    result = verifier.feedback()

    assert _report_json(result)["verification_id"] == "000"
    assert _report_json(result)["status"] == "REJECTED"
    assert _report_json(result)["executor_error"] == "model.py must be valid UTF-8"
    assert not (_coding_attempt(tmp_path) / "model.py").exists()
    assert len(executor.calls) == 1


def test_the_report_keeps_the_source_that_feedback_leaves_out(tmp_path: Path) -> None:
    """`verify` is what the workflow stores; `feedback` is what a model reads."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    report = verifier.verify()

    assert report.verification_id == "000"
    assert report.host_verification_dir == _coding_attempt(tmp_path)
    assert report.sandbox_verification_dir == "/work/attempts/round_000/coding/000"
    assert report.exec_report == replace(
        executor.report, step_path=_coding_attempt(tmp_path) / "output.step"
    )
    assert set(report.render_report) == {RESULT_NAME}
    assert report.drawing_diff_report is None
    assert "source" not in _report_json(verifier.feedback())


@pytest.mark.parametrize(
    "output_dirname",
    [
        PurePosixPath(""),
        PurePosixPath("."),
        PurePosixPath(".."),
        PurePosixPath("../attempts"),
        PurePosixPath("nested/attempts"),
        PurePosixPath("/work/attempts"),
    ],
)
def test_rejects_output_dirname_outside_workdir_root(
    tmp_path: Path,
    output_dirname: PurePosixPath,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="attempt root must be a directory basename"):
        _create_verifier(executor, workdir, output_dirname=output_dirname)


def test_rejects_symlink_output_directory(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (tmp_path / "attempts").symlink_to(outside_dir, target_is_directory=True)
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)

    with pytest.raises(ValueError, match="attempt root must not be a symlink"):
        _create_verifier(executor, workdir)


def test_verified_output_is_rendered_and_offered_to_the_model(tmp_path: Path) -> None:
    """A verified STEP must yield a drawing plus one render per style, and the
    tool result must point the model at all of them in sandbox coordinates."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        output_renders="path",
    )

    text = _text(verifier.feedback())

    verification_dir = _coding_attempt(tmp_path)
    (rendered_step, _, _, _) = renderer.calls[0]
    assert rendered_step == verification_dir / "output.step"
    sandbox_dir = f"{workdir.sandbox_bind_dir}/attempts/round_000/coding/000"
    for view in VIEWS:
        assert f"{sandbox_dir}/projection/{view}.dxf" in text
    # One pictorial of the one camera, whatever the renderer wrote.
    assert f"{sandbox_dir}/render_3d/{FEEDBACK_PICTORIAL}.png" in text
    for style in RENDER3D_STYLES:
        if style != FEEDBACK_PICTORIAL:
            assert style not in text


def test_rendered_artifacts_stay_inside_the_verification_directory(
    tmp_path: Path,
) -> None:
    """Nothing may be written where another attempt, or the agent, could see it."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=StubRenderer(),
        output_renders="path",
    )

    verifier.feedback()

    verification_dir = _coding_attempt(tmp_path)
    written = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {
        tmp_path / "model.py",
        verification_dir / "model.py",
        verification_dir / "output.step",
        *(verification_dir / "projection" / f"{view}.dxf" for view in VIEWS),
        *(verification_dir / "projection" / f"{view}.png" for view in VIEWS),
        *(verification_dir / "render_3d" / f"{style}.png" for style in RENDER3D_STYLES),
    }


def test_failed_verification_renders_nothing_and_reports_only_the_error(
    tmp_path: Path,
) -> None:
    """Without a valid STEP there is nothing to draw, so the renderer never runs."""
    executor = StubCadQueryExecutor(
        _execution_report(status=ExecutionStatus.FAILED, returncode=1)
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        output_renders="path",
    )

    result = verifier.feedback()

    assert renderer.calls == []
    assert _report_json(result)["status"] == "FAILED"
    assert "projection/front.dxf" not in _text(result)


def test_a_projection_that_cannot_be_read_is_explained_not_raised(
    tmp_path: Path,
) -> None:
    """A drawing on disk that will not open is a failed one, not a lost report."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=StubRenderer(corrupt_views=("front",)),
        output_renders="path",
    )

    text = _text(verifier.feedback())

    assert "- view_projected_front: unavailable" in text
    assert "view_projected_top (top)" in text


def test_a_pictorial_that_failed_is_explained_where_it_would_have_been(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer(skip_styles=(FEEDBACK_PICTORIAL,))
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        output_renders="path",
    )

    result = verifier.feedback()
    text = _text(result)

    assert f"{FEEDBACK_PICTORIAL}.png" not in text
    # The reason belongs where the render would have been, not in the report.
    assert "render_errors" not in _report_json(result)
    assert (
        f"- view_projected_{FEEDBACK_PICTORIAL}: unavailable "
        f"(RuntimeError: {FEEDBACK_PICTORIAL} failed)"
    ) in text


def test_a_style_that_is_not_offered_is_neither_named_nor_explained(
    tmp_path: Path,
) -> None:
    """The renderer draws three; a message that named all three would spend
    itself saying the same camera three times."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer(skip_styles=("transparent_shaded_edges_perspective",))
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        output_renders="path",
    )

    text = _text(verifier.feedback())

    assert "transparent_shaded_edges_perspective" not in text
    assert f"{FEEDBACK_PICTORIAL}.png" in text


def test_result_carries_paths_but_never_the_drawing_itself(
    tmp_path: Path,
) -> None:
    """The model is handed a path to open deliberately, not the DXF body."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=StubRenderer(),
        output_renders="path",
    )

    text = _text(verifier.feedback())

    dxf_body = (_coding_attempt(tmp_path) / "projection" / "front.dxf").read_text(
        encoding="utf-8"
    )
    assert "projection/front.dxf" in text
    assert dxf_body not in text


def test_images_are_embedded_only_when_the_presenter_asks_for_them(
    tmp_path: Path,
) -> None:
    """The feedback presentation mode decides whether images are embedded."""
    executor = StubCadQueryExecutor(_execution_report())
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")

    def block_types(mode: Literal["path", "image"]) -> list[str]:
        workdir = SandboxWorkdir(host_bind_dir=tmp_path)
        result = _create_verifier(
            executor,
            workdir,
            renderer=StubRenderer(),
            output_renders=mode,
        ).feedback()
        assert isinstance(result, list)
        return [block["type"] for block in result]

    assert "image" not in block_types("path")
    assert "image" in block_types("image")


def test_without_image_attachments_the_model_still_sees_orthographic_pngs(
    tmp_path: Path,
) -> None:
    """Both A/B conditions offer the same ordinary projections to inspect."""
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(
        executor, workdir, renderer=StubRenderer(), output_renders="path"
    )

    result = verifier.feedback()

    assert _report_json(result)["status"] == "VERIFIED"
    assert "projection/front.dxf" in _text(result)
    assert "/work/attempts/round_000/coding/000/projection/front.png" in _text(result)
    assert (_coding_attempt(tmp_path) / "projection" / "front.dxf").is_file()


def test_a_sealed_void_is_reported_with_its_place(tmp_path: Path) -> None:
    import cadquery as cq

    hollow = (
        cq.Workplane()
        .box(20, 20, 20)
        .val()
        .cut(cq.Solid.makeBox(4, 4, 4, cq.Vector(-2, -2, -2)))
    )
    cq.exporters.export(hollow, str(tmp_path / "part.step"), exportType="STEP")
    census = read_census(tmp_path / "part.step")
    assert census is not None
    void = "sealed voids 1 (cavities no opening reaches, bbox [xmin, ymin, zmin, xmax, ymax, zmax]: 64.0 mm³ at [-2.0, -2.0, -2.0, 2.0, 2.0, 2.0])"

    assert f"bbox 20.00 x 20.00 x 20.00; {void}; faces" in census.describe()
    assert "sealed voids" not in replace(census, voids=()).describe()


@pytest.mark.parametrize(
    "status", [ExecutionStatus.TIMEOUT, ExecutionStatus.INFRA_ERROR]
)
def test_a_build_the_machine_spoiled_is_attempted_again(
    status: ExecutionStatus, tmp_path: Path
) -> None:
    """Neither outcome describes the program, so the same bytes are built again."""
    executor = StubCadQueryExecutor(_execution_report(status=status, returncode=None))
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    verifier.verify()
    verifier.verify()

    assert len(executor.calls) == 2


def test_a_build_the_program_broke_is_not_attempted_again(tmp_path: Path) -> None:
    """A program that fails on its own fails the same way, so the build is kept."""
    executor = StubCadQueryExecutor(
        _execution_report(status=ExecutionStatus.FAILED, returncode=1)
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    verifier.verify()
    verifier.verify()

    assert len(executor.calls) == 1


@pytest.mark.parametrize(
    ("status", "source", "ready"),
    [
        (ExecutionStatus.VERIFIED, VALID_SOURCE, True),
        (ExecutionStatus.FAILED, VALID_SOURCE, False),
        (ExecutionStatus.REJECTED, "broken syntax (", False),
        (ExecutionStatus.REJECTED, None, False),
        (ExecutionStatus.TIMEOUT, VALID_SOURCE, False),
        (ExecutionStatus.INFRA_ERROR, VALID_SOURCE, False),
    ],
)
def test_only_verified_program_outcomes_are_confirmed(
    tmp_path: Path, status: ExecutionStatus, source: str | None, ready: bool
) -> None:
    executor = StubCadQueryExecutor(_execution_report(status=status, source=source))
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)
    assert not verifier.confirmed
    verifier.feedback()
    assert verifier.confirmed is ready
    assert verifier.verify().exec_report.status is status
    verifier.reset()
    assert not verifier.confirmed


def test_a_failed_build_blocks_only_by_its_status(tmp_path: Path) -> None:
    executor = StubCadQueryExecutor(
        _execution_report(status=ExecutionStatus.FAILED, returncode=1)
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)

    verifier.feedback()

    assert verifier.blockers == [
        "its build ended with status FAILED; the build report says why"
    ]


def test_failed_coding_submission_is_refused_after_feedback(tmp_path: Path) -> None:
    from langchain_core.messages import HumanMessage

    from tests.zeroshot.chat_models import ScriptedChatModel, tool_call
    from tests.zeroshot.workflow.test_agent import _subgraph
    from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware

    executor = StubCadQueryExecutor(
        _execution_report(status=ExecutionStatus.FAILED, returncode=1)
    )
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    verifier = _create_verifier(executor, workdir)
    answer = {
        "stage_report": {
            "dimension_checks": {},
            "concerns": {},
            "unticketed_changes": {},
        },
        "responses": {
            "ticket_initial": "The program still fails; audit must diagnose the operation."
        },
    }
    model = ScriptedChatModel(
        responses=(
            tool_call("TicketAnswers", answer, "first"),
            tool_call("TicketAnswers", answer, "after-feedback"),
        )
    )
    agent = _subgraph(
        model,
        tools=(),
        output_schema=TicketAnswers,
        response_format_strategy="tool",
        max_turns=2,
        extra_middleware=[VerifyOnWriteMiddleware(verifier)],
    )
    result = agent.invoke({"messages": [HumanMessage(content="Submit the program.")]})
    assert result.get("structured_response") is None
    assert len(model.received_messages) == 2
    assert any("FAILED" in m.text for m in model.received_messages[1])
    assert verifier.verify().exec_report.status is ExecutionStatus.FAILED
    assert len(executor.calls) == 1


def test_the_verifier_asks_for_exactly_the_views_it_was_given(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        views={view: UNTURNED[view] for view in (View.LEFT, View.BOTTOM)},
        output_renders="path",
    )

    text = _text(verifier.feedback())

    (_, projection_paths, _, frames) = renderer.calls[0]
    assert set(projection_paths.as_mapping()) == {"left", "bottom"}
    assert frames == {view: UNTURNED[view] for view in (View.LEFT, View.BOTTOM)}
    sandbox_dir = f"{workdir.sandbox_bind_dir}/attempts/round_000/coding/000"
    assert f"{sandbox_dir}/projection/left.dxf" in text
    assert "view_projected_left (left)" in text
    assert "front.dxf" not in text


def test_standard_projections_draw_all_six_views_without_an_interpretation(
    tmp_path: Path,
) -> None:
    executor = StubCadQueryExecutor(_execution_report())
    workdir = SandboxWorkdir(host_bind_dir=tmp_path)
    (tmp_path / "model.py").write_text(VALID_SOURCE, encoding="utf-8")
    renderer = StubRenderer()
    verifier = _create_verifier(
        executor,
        workdir,
        renderer=renderer,
        output_renders="path",
        projection_view_mode="standard",
    )

    text = _text(verifier.feedback())

    assert verifier.interpretation is None
    (_, projection_paths, _, frames) = renderer.calls[0]
    assert frames == STANDARD_VIEW_FRAMES
    assert set(projection_paths.as_mapping()) == set(STANDARD_VIEW_FRAMES)
    sandbox_dir = f"{workdir.sandbox_bind_dir}/attempts/round_000/coding/000"
    for view in STANDARD_VIEW_FRAMES:
        # Listed, not unavailable: each sheet registered under its own axes.
        assert (
            f"- view_projected_{view} ({view}): {sandbox_dir}/projection/{view}.dxf"
            in text
        )
        assert f"{sandbox_dir}/projection/{view}.png" in text


@pytest.mark.parametrize(
    ("projection_view_mode", "diff_drawer"),
    [("third_angle", None), ("standard", object())],
)
def test_a_projection_view_mode_the_verifier_cannot_serve_is_refused(
    tmp_path: Path, projection_view_mode: str, diff_drawer: object | None
) -> None:
    with pytest.raises(ValueError, match="projection_view_mode"):
        _create_verifier(
            StubCadQueryExecutor(_execution_report()),
            SandboxWorkdir(host_bind_dir=tmp_path),
            projection_view_mode=projection_view_mode,  # type: ignore[arg-type]
            diff_drawer=diff_drawer,
        )


# An L-shaped block in the positive octant, so no two standard views share extents.
ASYMMETRIC_SOURCE = """\
import cadquery as cq

block = cq.Workplane("XY").box(30, 20, 10, centered=False)
notch = cq.Workplane("XY").box(12, 20, 5, centered=False).translate((18, 0, 5))
result = block.cut(notch)
"""

# (u_min, v_min, u_max, v_max) of each standard view of that block.
ASYMMETRIC_EXTENTS = {
    "front": (0, 0, 30, 10),
    "back": (-30, 0, 0, 10),
    "top": (0, 0, 30, 20),
    "bottom": (0, -20, 30, 0),
    "left": (-20, 0, 0, 10),
    "right": (0, 0, 20, 10),
}


def test_real_cadquery_render_reaches_feedback_in_six_standard_views(
    tmp_path: Path,
) -> None:
    import sys

    from zeroshot.pipeline.sandbox import SandboxRunner
    from zeroshot.pipeline.verification import CadQueryExecutor, StepRenderer

    (tmp_path / "model.py").write_text(ASYMMETRIC_SOURCE)
    with SandboxWorkdir(host_bind_dir=tmp_path) as workdir:
        verifier = OutputVerifier(
            executor=CadQueryExecutor(
                sandbox_runner=SandboxRunner(
                    python_executable=Path(sys.executable), default_timeout_s=30
                )
            ),
            workdir=workdir,
            renderer=StepRenderer(max_workers=1),
            diff_drawer=None,
            artifact_presenter=ArtifactPresenter(input="path"),
            attempt_store=AttemptStore(workdir, round_source=lambda: 0),
            projection_view_mode="standard",
        )
        text = _text(verifier.feedback())
        report = verifier.verify()

    assert report.exec_report.status is ExecutionStatus.VERIFIED
    render = report.render_report[RESULT_NAME]
    assert render.status is RenderStatus.OK
    projections = render.projection_paths.as_mapping()
    assert set(projections) == set(ASYMMETRIC_EXTENTS)
    sandbox_dir = "/work/attempts/round_000/coding/000"
    for view, extents in ASYMMETRIC_EXTENTS.items():
        header = ezdxf.readfile(projections[view]).header
        low, high = header["$EXTMIN"], header["$EXTMAX"]
        assert (low[0], low[1], high[0], high[1]) == pytest.approx(extents, abs=1e-3)
        with Image.open(projections[view].with_suffix(".png")) as png:
            assert png.convert("L").getextrema()[0] < 128, f"{view}.png is blank"
        assert f"{sandbox_dir}/projection/{view}.png" in text
    assert render.render3d_paths.hlg_translucent_faces_perspective.is_file()
    assert f"{sandbox_dir}/render_3d/{FEEDBACK_PICTORIAL}.png" in text


@pytest.mark.parametrize(
    ("kind", "mode"),
    [
        (kind, mode)
        for kind in ("output_renders", "overlay", "unmatched")
        for mode in (
            ("path", "image")
            if kind in {"output_renders", "unmatched"}
            else ("none", "path", "image")
        )
    ],
)
def test_feedback_kinds_are_presented_independently(tmp_path, kind, mode):
    (tmp_path / "model.py").write_text(VALID_SOURCE)
    modes = {
        "output_renders": "path",
        "unmatched": "path",
        "overlay": "none",
    }
    modes[kind] = mode
    verifier = _create_verifier(
        StubCadQueryExecutor(_execution_report()),
        SandboxWorkdir(tmp_path),
        diff_drawer=StubDiffDrawer(chamfers=[6.0]),
        **modes,
    )
    _set_input_crop(verifier, tmp_path)
    blocks = verifier.feedback()
    text = _text(blocks)
    paths = {
        "output_renders": "/coding/000/projection/front.png",
        "overlay": "front_overlay.png",
        "unmatched": "front_unmatched.png",
    }
    for name, path in paths.items():
        assert (path in text) == (modes[name] != "none")
    for name, legend in {
        "overlay": "Overlay images: input lines in pale gray",
        "unmatched": "hatching marks material (silhouette) differences",
    }.items():
        assert (legend in text) == (modes[name] != "none")
    assert "[Drawing comparison]" in text
    assert "line distance" in text
    expected_images = (2 if kind == "output_renders" else 1) if mode == "image" else 0
    assert sum(block["type"] == "image" for block in blocks) == expected_images
    # Presentation never suppresses execution or final STEP rendering.
    assert verifier.confirmed
    assert RESULT_NAME in verifier.verify().render_report
