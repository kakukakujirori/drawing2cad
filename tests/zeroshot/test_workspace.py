from collections.abc import Iterator
from pathlib import Path

import pytest

from zeroshot.pipeline.manifest import SampleManifest
from zeroshot.pipeline.workspace import Workspace, create_workspace

RENDER_STYLES = (
    "hlg_perspective",
    "transparent_shaded_edges_perspective",
    "hlg_translucent_faces_perspective",
)
SAMPLE_ID = "000364"


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def manifest(tmp_path: Path) -> SampleManifest:
    dataset_root = tmp_path / "test_vlm"
    return SampleManifest(
        sample_id=SAMPLE_ID,
        input_dxf_path=_write(
            dataset_root / "techdraw" / "dxf" / f"{SAMPLE_ID}.dxf",
            b"DXF",
        ),
        input_render3d_paths={
            style: _write(
                dataset_root / "render_3d" / style / f"{SAMPLE_ID}.png",
                f"image:{style}".encode(),
            )
            for style in RENDER_STYLES
        },
    )


@pytest.fixture
def workspace(
    manifest: SampleManifest,
) -> Iterator[Workspace]:
    with create_workspace(
        manifest=manifest,
        render_styles=RENDER_STYLES,
    ) as created:
        yield created


def test_stages_inputs_and_returns_workspace_relative_paths(
    workspace: Workspace,
) -> None:
    assert workspace.input_dxf_relpath == Path("input/input.dxf")
    assert not workspace.input_dxf_relpath.is_absolute()
    assert (
        workspace.host_bind_dir / workspace.input_dxf_relpath
    ).read_bytes() == b"DXF"

    assert set(workspace.input_render3d_relpaths) == set(RENDER_STYLES)
    for style, relpath in workspace.input_render3d_relpaths.items():
        assert relpath == Path("input") / f"{style}.png"
        assert not relpath.is_absolute()
        assert (
            workspace.host_bind_dir / relpath
        ).read_bytes() == f"image:{style}".encode()


def test_stages_only_selected_render_styles(
    manifest: SampleManifest,
) -> None:
    selected_style = RENDER_STYLES[0]

    with create_workspace(
        manifest=manifest,
        render_styles=[selected_style],
    ) as workspace:
        assert workspace.input_render3d_relpaths == {
            selected_style: Path("input") / f"{selected_style}.png"
        }
        assert not (
            workspace.host_bind_dir / "input" / f"{RENDER_STYLES[1]}.png"
        ).exists()


def test_staged_files_are_independent_copies(
    manifest: SampleManifest,
) -> None:
    selected_style = RENDER_STYLES[0]

    with create_workspace(
        manifest=manifest,
        render_styles=[selected_style],
    ) as workspace:
        staged_dxf = workspace.host_bind_dir / workspace.input_dxf_relpath
        staged_render = (
            workspace.host_bind_dir
            / workspace.input_render3d_relpaths[selected_style]
        )
        staged_dxf.write_bytes(b"changed DXF")
        staged_render.write_bytes(b"changed image")

        assert manifest.input_dxf_path.read_bytes() == b"DXF"
        assert (
            manifest.input_render3d_paths[selected_style].read_bytes()
            == f"image:{selected_style}".encode()
        )


def test_removes_workspace_after_context_exit(
    manifest: SampleManifest,
) -> None:
    with create_workspace(
        manifest=manifest,
        render_styles=[],
    ) as workspace:
        bind_dir = workspace.host_bind_dir
        assert bind_dir.is_dir()

    assert not bind_dir.exists()


def test_rejects_duplicate_render_styles(
    manifest: SampleManifest,
) -> None:
    with (
        pytest.raises(ValueError, match="Duplicate render styles"),
        create_workspace(
            manifest=manifest,
            render_styles=[RENDER_STYLES[0], RENDER_STYLES[0]],
        ),
    ):
        pass


def test_rejects_unknown_render_style(
    manifest: SampleManifest,
) -> None:
    with (
        pytest.raises(ValueError, match="superset"),
        create_workspace(
            manifest=manifest,
            render_styles=["unknown"],
        ),
    ):
        pass


def test_render_path_mapping_is_immutable(
    workspace: Workspace,
) -> None:
    with pytest.raises(TypeError):
        workspace.input_render3d_relpaths["new-style"] = Path("input/new.png")  # type: ignore[index]
