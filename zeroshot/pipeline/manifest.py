from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class SampleManifest:
    sample_id: str
    input_dxf_path: Path
    input_render3d_paths: Mapping[str, Path]
    feedback_dxf_path: Path | None = None
    feedback_render3d_paths: Mapping[str, Path] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = self.sample_id.strip()
        if not sample_id:
            raise ValueError("sample_id must not be empty")
        if sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
            raise ValueError(f"unsafe sample_id: {sample_id!r}")

        # str -> Path | None
        input_dxf_path = Path(self.input_dxf_path)
        input_render3d_paths = MappingProxyType(
            {style: Path(path) for style, path in self.input_render3d_paths.items()}
        )
        feedback_dxf_path = (
            Path(self.feedback_dxf_path) if self.feedback_dxf_path is not None else None
        )
        feedback_render3d_paths = MappingProxyType(
            {style: Path(path) for style, path in self.feedback_render3d_paths.items()}
        )

        # input sanity check
        if not input_dxf_path.is_file():
            raise FileNotFoundError(f"Not Found: {self.input_dxf_path}")
        if input_dxf_path.suffix.lower() != ".dxf":
            raise ValueError(f"DXF path must end in .dxf: {self.input_dxf_path}")
        for path in input_render3d_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Not Found: {path}")

        # feedback sanity check
        if feedback_dxf_path is not None:
            if not feedback_dxf_path.is_file():
                raise FileNotFoundError(f"Not Found: {self.feedback_dxf_path}")
            if feedback_dxf_path.suffix.lower() != ".dxf":
                raise ValueError(f"DXF path must end in .dxf: {self.feedback_dxf_path}")
        if feedback_render3d_paths:
            for path in feedback_render3d_paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"Not Found: {path}")

        # feedback styles <= input styles
        if not set(feedback_render3d_paths.keys()) <= set(input_render3d_paths.keys()):
            raise ValueError(
                "Feedback styles must be a subset of input styles: "
                f"feedback={set(feedback_render3d_paths.keys())}, "
                f"input={set(input_render3d_paths.keys())}"
            )

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "input_dxf_path", input_dxf_path)
        object.__setattr__(self, "input_render3d_paths", input_render3d_paths)
        object.__setattr__(self, "feedback_dxf_path", feedback_dxf_path)
        object.__setattr__(self, "feedback_render3d_paths", feedback_render3d_paths)

    def load_input_render3d(self, styles: list[str]) -> Mapping[str, bytes]:
        return {
            style: self.input_render3d_paths[style].read_bytes() for style in styles
        }

    def load_feedback_render3d(self, styles: list[str]) -> Mapping[str, bytes]:
        return {
            style: self.feedback_render3d_paths[style].read_bytes() for style in styles
        }
