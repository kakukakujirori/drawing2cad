"""Each audit finding arrives at its ticket with pictures of what it measured."""

from collections.abc import Iterator
from pathlib import Path

import ezdxf
import pytest
from PIL import Image, ImageChops

from tests.zeroshot.workflow.test_audit_drawings import cite, report
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.verification.render.export_dxf import export_to_png
from zeroshot.pipeline.workflow.evidence import _pixels_of, render_evidence


@pytest.fixture
def workspace() -> Iterator[SandboxWorkdir]:
    """A 40x20 raster and a DXF of the same rectangle in millimetres."""
    with SandboxWorkdir() as workdir:
        Image.new("RGB", (40, 20), "white").save(workdir.host_bind_dir / "front.png")
        projection = workdir.host_bind_dir / "projection"
        projection.mkdir()
        document = ezdxf.new()
        document.modelspace().add_lwpolyline(
            [(0, 0), (40, 0), (40, 20), (0, 20)], close=True
        )
        document.saveas(projection / "front.dxf")
        yield workdir


def _finding(*regions):
    return report(cites=list(regions), target="sem_bore").findings[0]


def test_a_raster_region_is_cut_out_at_its_own_pixels(
    workspace: SandboxWorkdir,
) -> None:
    written = render_evidence(
        _finding(cite("front.png", (5, 4, 25, 16))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
        mode="crop",
    )

    assert written == ["/work/tickets/ticket_001_bore/evidence_0.png"]
    with Image.open(workspace.sandbox_to_host_path(written[0])) as crop:
        assert crop.size == (20, 12)


def test_a_dxf_region_is_rasterised_and_cut_out_at_its_millimetres(
    workspace: SandboxWorkdir,
) -> None:
    """The left half, measured up from the bottom as UV does."""
    written = render_evidence(
        _finding(cite("projection/front.dxf", (0.0, 0.0, 20.0, 20.0))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
        mode="crop",
    )

    crop_path = workspace.sandbox_to_host_path(written[0])
    with Image.open(crop_path) as crop:
        width, height = crop.size
    assert width == pytest.approx(height, rel=0.02), "a square half of a 40x20 sheet"
    assert not list(crop_path.parent.glob("*_sheet.png")), (
        "the whole sheet is temporary"
    )


def test_every_region_keeps_the_findings_order(workspace: SandboxWorkdir) -> None:
    written = render_evidence(
        _finding(
            cite("projection/front.dxf", (0.0, 0.0, 20.0, 20.0)),
            cite("front.png", (0, 0, 10, 10)),
        ),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
    )

    assert [Path(path).name for path in written] == [
        "evidence_0.png",
        "evidence_1.png",
    ]


@pytest.mark.parametrize("margin_ratio", [0.0, 0.04, 0.2])
def test_a_dxf_crop_holds_the_shape_its_region_surrounds(
    workspace: SandboxWorkdir,
    margin_ratio: float,
) -> None:
    """Raster margins must not shift a region away from its DXF coordinates."""
    source = workspace.host_bind_dir / "projection" / "front.dxf"
    document = ezdxf.readfile(source)
    document.modelspace().add_circle((2, 10), 1.0)
    document.saveas(source)

    written = render_evidence(
        _finding(cite("projection/front.dxf", (0.5, 8.5, 3.5, 11.5))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
        margin_ratio=margin_ratio,
        mode="crop",
    )

    with Image.open(workspace.sandbox_to_host_path(written[0])) as crop:
        darkest = min(crop.convert("L").tobytes())  # one byte per grey pixel
        ink = ImageChops.invert(crop.convert("L")).point(
            lambda p: 255 if p > 127 else 0
        )
        left, top, right, bottom = ink.getbbox()
        assert (left + right) / 2 == pytest.approx(crop.width / 2, abs=1)
        assert (top + bottom) / 2 == pytest.approx(crop.height / 2, abs=1)
    assert darkest < 128, "the circle the region surrounds is missing from the crop"


@pytest.mark.parametrize("margin_ratio", [0.04, 0.2])
def test_dxf_box_matches_rendered_edges_with_margins(tmp_path, margin_ratio):
    source = tmp_path / "offset.dxf"
    document = ezdxf.new()
    document.modelspace().add_lwpolyline(
        [(-30, 10), (10, 10), (10, 30), (-30, 30)], close=True
    )
    document.saveas(source)
    with Image.open(export_to_png(source, margin_ratio=margin_ratio)) as image:
        ink = ImageChops.invert(image.convert("L")).point(
            lambda p: 255 if p > 127 else 0
        )
        box = _pixels_of(
            cite(str(source), (-30, 10, 10, 30)),
            source,
            image.size,
            margin_ratio=margin_ratio,
        )
        assert box == pytest.approx(ink.getbbox(), abs=1)


def test_a_single_pixel_region_makes_a_picture(
    workspace: SandboxWorkdir,
) -> None:
    written = render_evidence(
        _finding(cite("front.png", (5, 5, 6, 6))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
        mode="crop",
    )

    with Image.open(workspace.sandbox_to_host_path(written[0])) as crop:
        assert crop.size == (1, 1)


@pytest.mark.parametrize("box", [(5, 4, 25, 16), (5, 5, 6, 6)])
def test_default_mark_keeps_raster_context_and_source(workspace, box):
    source = workspace.host_bind_dir / "front.png"
    original = source.read_bytes()
    written = render_evidence(
        _finding(cite("front.png", box)), workspace.host_bind_dir / "marked",
        workspace,
    )
    with Image.open(workspace.sandbox_to_host_path(written[0])) as image:
        assert image.size == (40, 20)
        assert image.getpixel(box[:2]) == (255, 0, 0)
        assert image.getpixel((box[2] - 1, box[3] - 1)) == (255, 0, 0)
        assert image.getpixel((0, 0)) == (255, 255, 255)
    assert source.read_bytes() == original


@pytest.mark.parametrize("margin_ratio", [0.04, 0.2])
def test_mark_uses_padded_dxf_frame_and_flips_vertical_axis(workspace, margin_ratio):
    source = workspace.host_bind_dir / "projection/front.dxf"
    region = cite("projection/front.dxf", (10, 2, 30, 8))
    sheet = export_to_png(source, margin_ratio=margin_ratio)
    with Image.open(sheet) as image:
        size = image.size
    written = render_evidence(
        _finding(region), workspace.host_bind_dir / "marked", workspace,
        mode="mark", margin_ratio=margin_ratio,
    )
    with Image.open(workspace.sandbox_to_host_path(written[0])) as image:
        assert image.size == size
        box = _pixels_of(region, source, size, margin_ratio=margin_ratio)
        assert box[1] > size[1] / 2
        assert image.getpixel(box[:2]) == (255, 0, 0)
        assert image.getpixel((box[2] - 1, box[3] - 1)) == (255, 0, 0)
        assert image.getpixel((size[0] // 2, size[1] // 4)) == (255, 255, 255)
    assert not list((workspace.host_bind_dir / "marked").glob("*_sheet.png"))
