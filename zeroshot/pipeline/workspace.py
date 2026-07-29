import shutil
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from zeroshot.pipeline.manifest import SampleManifest


@dataclass(frozen=True)
class Workspace:
    host_bind_dir: Path
    input_dxf_relpath: Path
    input_render3d_relpaths: Mapping[str, Path]


@contextmanager
def create_workspace(
    manifest: SampleManifest,
    render_styles: Sequence[str],
) -> Generator[Workspace, None, None]:

    # sanity check
    if len(set(render_styles)) != len(render_styles):
        raise ValueError(f"Duplicate render styles: {render_styles}")

    if not set(render_styles) <= set(manifest.input_render3d_paths.keys()):
        raise ValueError(
            f"input render3d styles must be a superset of render styles: "
            f"{render_styles=}, {manifest.input_render3d_paths.keys()=}"
        )

    with tempfile.TemporaryDirectory(
        prefix=f"drawing2cad-{manifest.sample_id}-"
    ) as tmp_dir:
        host_bind_dir = Path(tmp_dir)

        # Copy inputs to host_bind_dir
        workspace_input_relpath = Path("input")
        workspace_input_dir = host_bind_dir / workspace_input_relpath
        workspace_input_dir.mkdir(parents=True, exist_ok=False)

        # DXF
        workspace_dxf_relpath = workspace_input_relpath / "input.dxf"
        shutil.copyfile(
            manifest.input_dxf_path,
            host_bind_dir / workspace_dxf_relpath,
        )

        # Perspective renders
        workspace_render3d_relpaths = {}
        for style in render_styles:
            workspace_render3d_relpaths[style] = workspace_input_relpath / f"{style}{manifest.input_render3d_paths[style].suffix}"
            shutil.copyfile(
                manifest.input_render3d_paths[style],
                host_bind_dir / workspace_render3d_relpaths[style]
            )

        yield Workspace(
            host_bind_dir=host_bind_dir,
            input_dxf_relpath=workspace_dxf_relpath,
            input_render3d_relpaths=MappingProxyType(workspace_render3d_relpaths),
        )
