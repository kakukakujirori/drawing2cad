from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class InputManifest:
    sample_id: str
    dxf_path: Path
    render3d_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        sample_id = self.sample_id.strip()
        if not sample_id:
            raise ValueError("sample_id must not be empty")
        if sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id:
            raise ValueError(f"unsafe sample_id: {sample_id!r}")

        # str -> Path
        dxf_path = Path(self.dxf_path)
        render3d_paths = MappingProxyType(
            {style: Path(path) for style, path in self.render3d_paths.items()}
        )

        # sanity check
        if not dxf_path.is_file():
            raise FileNotFoundError(f"Not Found: {self.dxf_path}")
        if dxf_path.suffix.lower() != ".dxf":
            raise ValueError(f"DXF path must end in .dxf: {self.dxf_path}")
        for path in render3d_paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Not Found: {path}")

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "dxf_path", dxf_path)
        object.__setattr__(self, "render3d_paths", render3d_paths)

    def load_render3d(self, styles: list[str]) -> Mapping[str, bytes]:
        return {style: self.render3d_paths[style].read_bytes() for style in styles}


@dataclass(frozen=True)
class FeedbackManifest:
    verification_id: str
    dxf_path: Path | None = None
    dxf_error: str | None = None
    render3d_paths: Mapping[str, Path] = field(default_factory=dict)
    render3d_errors: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        verification_id = self.verification_id.strip()
        if not verification_id:
            raise ValueError("verification_id must not be empty")
        if (
            verification_id in {".", ".."}
            or "/" in verification_id
            or "\\" in verification_id
        ):
            raise ValueError(f"unsafe verification_id: {verification_id!r}")

        # str | None -> Path | None
        dxf_path = Path(self.dxf_path) if self.dxf_path is not None else None
        render3d_paths = MappingProxyType(
            {style: Path(path) for style, path in self.render3d_paths.items()}
        )
        render3d_errors = MappingProxyType(dict(self.render3d_errors))

        # sanity check
        if dxf_path is not None:
            if not dxf_path.is_file():
                raise FileNotFoundError(f"Not Found: {self.dxf_path}")
            if dxf_path.suffix.lower() != ".dxf":
                raise ValueError(f"DXF path must end in .dxf: {self.dxf_path}")
        if render3d_paths:
            for path in render3d_paths.values():
                if not path.is_file():
                    raise FileNotFoundError(f"Not Found: {path}")

        # an artifact is either present or explained, never both
        if dxf_path is not None and self.dxf_error is not None:
            raise ValueError(f"dxf_path and dxf_error are both set: {self.dxf_error!r}")
        both = sorted(set(render3d_paths) & set(render3d_errors))
        if both:
            raise ValueError(f"styles are both rendered and failed: {both}")

        # re-register
        object.__setattr__(self, "verification_id", verification_id)
        object.__setattr__(self, "dxf_path", dxf_path)
        object.__setattr__(self, "render3d_paths", render3d_paths)
        object.__setattr__(self, "render3d_errors", render3d_errors)

    def load_render3d(self, styles: list[str]) -> Mapping[str, bytes]:
        return {style: self.render3d_paths[style].read_bytes() for style in styles}
