"""Each audit finding arrives at its ticket with pictures of what it measured."""

from collections.abc import Iterator
from pathlib import Path

import ezdxf
import pytest
from PIL import Image

from tests.zeroshot.workflow.test_audit_drawings import cite, report
from zeroshot.pipeline.sandbox import SandboxWorkdir
from zeroshot.pipeline.workflow.evidence import crop_evidence


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
    written = crop_evidence(
        _finding(cite("front.png", (5, 4, 25, 16))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
    )

    assert written == ["/work/tickets/ticket_001_bore/evidence_0.png"]
    with Image.open(workspace.sandbox_to_host_path(written[0])) as crop:
        assert crop.size == (20, 12)


def test_a_dxf_region_is_rasterised_and_cut_out_at_its_millimetres(
    workspace: SandboxWorkdir,
) -> None:
    """The left half, measured up from the bottom as UV does."""
    written = crop_evidence(
        _finding(cite("projection/front.dxf", (0.0, 0.0, 20.0, 20.0))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
    )

    crop_path = workspace.sandbox_to_host_path(written[0])
    with Image.open(crop_path) as crop:
        width, height = crop.size
    assert width == pytest.approx(height, rel=0.02), "a square half of a 40x20 sheet"
    assert not list(crop_path.parent.glob("*_sheet.png")), (
        "the whole sheet is temporary"
    )


def test_every_region_keeps_the_findings_order(workspace: SandboxWorkdir) -> None:
    written = crop_evidence(
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


def test_a_dxf_crop_holds_the_shape_its_region_surrounds(
    workspace: SandboxWorkdir,
) -> None:
    """The picture's frame is the drawing's own, not matplotlib's padded one."""
    source = workspace.host_bind_dir / "projection" / "front.dxf"
    document = ezdxf.readfile(source)
    document.modelspace().add_circle((2, 10), 1.0)
    document.saveas(source)

    written = crop_evidence(
        _finding(cite("projection/front.dxf", (0.5, 8.5, 3.5, 11.5))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
    )

    with Image.open(workspace.sandbox_to_host_path(written[0])) as crop:
        darkest = min(crop.convert("L").tobytes())  # one byte per grey pixel
    assert darkest < 128, "the circle the region surrounds is missing from the crop"


def test_a_region_thinner_than_a_pixel_still_makes_a_picture(
    workspace: SandboxWorkdir,
) -> None:
    written = crop_evidence(
        _finding(cite("front.png", (5.1, 5.1, 5.4, 5.4))),
        workspace.host_bind_dir / "tickets" / "ticket_001_bore",
        workspace,
    )

    with Image.open(workspace.sandbox_to_host_path(written[0])) as crop:
        assert crop.size == (1, 1)
