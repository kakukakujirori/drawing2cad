from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import cleandoc
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Literal

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.content import (
    ContentBlock,
    create_image_block,
    create_text_block,
)

from zeroshot.pipeline.manifest import FeedbackManifest, InputManifest
from zeroshot.pipeline.sandbox import SandboxWorkdir


@dataclass(frozen=True)
class _SandboxManifest:
    """Files exposed to the agent in the sandbox namespace."""

    id: str
    dxf_path: PurePosixPath | None
    render3d_paths: Mapping[str, PurePosixPath]
    render3d_bytes: Mapping[str, bytes]
    execution_feedback: str | None


@dataclass(frozen=True)
class MessageBuilder:
    access_render3d: Literal["none", "path", "image"]
    access_render3d_styles: tuple[str, ...]
    feedback_render3d: Literal["none", "path", "image"]
    feedback_render3d_styles: tuple[str, ...]
    system_prompt: Callable[[str, str], str] = lambda output_path, verification_dir: (
        cleandoc(
            f"""
        You are an expert CAD engineer specializing in reconstructing accurate parametric 3D models from engineering drawings.
        Convert the provided three-view DXF drawing into a valid 3D CAD model and write a complete, executable CadQuery Python script to:

        {output_path}

        Requirements:
        - The output CADQuery script must be self-contained and not load the input DXF or any external files.
        - Store the completed CADQuery solid in the `result` variable in the output file.
        - Use all available perspective renders together if provided.
        - The generated geometry must be valid and exportable to STEP.

        Tools:
        - `run_shell` can call any bash functions. Use it to read, write and inspect any files.
        - `load_image` loads the image data from the specified path.
        - `verify_output` compiles {output_path} and generates a STEP file and its 2D renderings. It returns error messages when STEP generation fails.

        Tips:
        - Create temporary Python scripts and run them using the `run_shell` tool with the command `python <filename>` for quick analysis and validation.
        - Use `ezdxf` Python library to analyze the input DXF file.
        - Use the `verify_output` results to debug and refine the {output_path} script.
        - Review the past `verify_output` results under {verification_dir} if necessary.
        """.strip()
            + "\n"
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "access_render3d_styles",
            tuple(self.access_render3d_styles),
        )
        object.__setattr__(
            self,
            "feedback_render3d_styles",
            tuple(self.feedback_render3d_styles),
        )
        self._validate_selection(
            "access_render3d",
            self.access_render3d,
            self.access_render3d_styles,
        )
        self._validate_selection(
            "feedback_render3d",
            self.feedback_render3d,
            self.feedback_render3d_styles,
        )

    @staticmethod
    def _validate_selection(name: str, mode: str, styles: tuple[str, ...]) -> None:
        if mode not in {"none", "path", "image"}:
            raise ValueError(f"invalid {name}: {mode!r}")
        if len(styles) != len(set(styles)):
            raise ValueError(f"duplicate styles in {name}: {styles}")
        if mode != "none" and not styles:
            raise ValueError(f"{name}_styles must not be empty when mode={mode!r}")

    @staticmethod
    def _translate_paths(
        manifest: InputManifest | FeedbackManifest,
        workdir: SandboxWorkdir,
        render3d_mode: Literal["none", "path", "image"],
    ) -> _SandboxManifest:

        # paths to translate
        paths_to_translate = dict(manifest.render3d_paths).copy()
        if manifest.dxf_path is not None:
            assert "dxf" not in paths_to_translate  # NEVER HAPPENS
            paths_to_translate["dxf"] = manifest.dxf_path

        # existence check
        for key, path in paths_to_translate.items():
            if path.is_symlink():
                raise ValueError(f"{path} must not be a symlink")
            if not path.is_absolute():
                raise ValueError(f"{path} must be absolute")
            if not path.is_relative_to(workdir.host_bind_dir):
                raise ValueError(f"{path} must be under {workdir.host_bind_dir}")
            if not path.is_file():
                raise FileNotFoundError(f"{key} not found: {path}")

        # translate paths
        translated_paths = {
            key: workdir.host_to_sandbox_path(path)
            for key, path in paths_to_translate.items()
        }
        translated_dxf_path = translated_paths.pop("dxf", None)
        translated_render3d_paths = MappingProxyType(translated_paths)

        # load images
        translated_render3d_bytes = (
            MappingProxyType(
                {
                    style: path.read_bytes()
                    for style, path in manifest.render3d_paths.items()
                }
            )
            if render3d_mode == "image"
            else MappingProxyType({})
        )

        # other fields
        manifest_id = (
            manifest.sample_id
            if isinstance(manifest, InputManifest)
            else manifest.verification_id
        )
        execution_feedback = (
            manifest.execution_feedback
            if isinstance(manifest, FeedbackManifest)
            else None
        )

        return _SandboxManifest(
            id=manifest_id,
            dxf_path=translated_dxf_path,
            render3d_paths=translated_render3d_paths,
            render3d_bytes=translated_render3d_bytes,
            execution_feedback=execution_feedback,
        )

    def build_initial(
        self,
        manifest: InputManifest,
        workdir: SandboxWorkdir,
        output_filename: str = "model.py",
        verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    ) -> list[BaseMessage]:
        """
        Build messages from files staged in workdir.host_bind_dir.
        Paths included in text are translated to workdir.sandbox_bind_dir paths.
        """
        self._validate_requested_styles(self.access_render3d_styles, manifest)
        self._validate_requested_styles(self.feedback_render3d_styles, manifest)

        sandbox_manifest = self._translate_paths(
            manifest, workdir, self.access_render3d
        )

        blocks: list[ContentBlock] = [
            create_text_block(
                "\n".join(
                    [
                        f"[Input DXF path: {sandbox_manifest.dxf_path}]",
                        "The DXF contains the three-view 2D technical drawing.",
                        "The coordinate system of the 2D drawing is as follows:",
                        "- Front view: right=+x, up=+y",
                        "- Top view: right=+x, up=-z",
                        "- Right view: right=-z, up=+y",
                        "",
                    ]
                )
            )
        ]

        blocks.extend(
            self._render_blocks(
                mode=self.access_render3d,
                render3d_styles=self.access_render3d_styles,
                render3d_paths=sandbox_manifest.render3d_paths,
                render3d_bytes=sandbox_manifest.render3d_bytes,
                label="[Input perspective renders]",
            )
        )

        output_path = str(workdir.sandbox_bind_dir / output_filename)
        verification_dir = str(workdir.sandbox_bind_dir / verification_dirname)
        return [
            SystemMessage(content=self.system_prompt(output_path, verification_dir)),
            HumanMessage(content_blocks=blocks),
        ]

    def build_feedback(
        self,
        manifest: FeedbackManifest,
        workdir: SandboxWorkdir,
    ) -> HumanMessage:
        sandbox_manifest = self._translate_paths(
            manifest, workdir, self.feedback_render3d
        )

        blocks: list[ContentBlock] = [
            create_text_block(
                f"[Candidate execution feedback]\n{sandbox_manifest.execution_feedback}\n"
            )
        ]

        if sandbox_manifest.dxf_path is not None:
            blocks.append(
                create_text_block(
                    f"[Projected DXF path: {sandbox_manifest.dxf_path}]\n"
                )
            )

        blocks.extend(
            self._render_blocks(
                mode=self.feedback_render3d,
                render3d_styles=self.feedback_render3d_styles,
                render3d_paths=sandbox_manifest.render3d_paths,
                render3d_bytes=sandbox_manifest.render3d_bytes,
                label="[Projected perspective renders]",
            )
        )

        blocks.append(
            create_text_block(
                "Use this feedback to revise the candidate, or submit a corrected final candidate."
            )
        )
        return HumanMessage(content_blocks=blocks)

    @staticmethod
    def _validate_requested_styles(
        styles: tuple[str, ...], manifest: InputManifest
    ) -> None:
        unknown_styles = set(styles) - set(manifest.render3d_paths)
        if unknown_styles:
            raise ValueError(f"Unknown styles specified: {sorted(unknown_styles)}")

    @staticmethod
    def _render_blocks(
        mode: Literal["none", "path", "image"],
        render3d_styles: tuple[str, ...],
        render3d_paths: Mapping[str, PurePosixPath],
        render3d_bytes: Mapping[str, bytes],
        label: str,
    ) -> list[ContentBlock]:

        available_styles = [s for s in render3d_styles if s in render3d_paths]

        if mode == "none" or not available_styles:
            return []

        elif mode == "path":
            return [
                create_text_block(
                    "\n".join(
                        [
                            f"{label}",
                            "\n".join(
                                f"- {style}: {render3d_paths[style]}"
                                for style in available_styles
                            ),
                            "",
                        ]
                    )
                )
            ]

        elif mode == "image":
            blocks: list[ContentBlock] = [
                create_text_block(f"{label}\nSee the attached images.\n")
            ]
            for style in available_styles:
                blocks.append(create_text_block(f"- {style}: {render3d_paths[style]}"))
                blocks.append(
                    create_image_block(
                        base64=base64.b64encode(render3d_bytes[style]).decode("ascii"),
                        mime_type="image/png",
                    )
                )
            return blocks

        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
