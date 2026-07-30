from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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

DEFAULT_SYSTEM_PROMPT = (
    """
You are an expert CAD engineer specializing in reconstructing parametric 3D models from engineering drawings.
Your task is to convert a three-view DXF drawing into a valid 3D CAD model and produce a complete, executable CadQuery Python script.

Note:
- In the output CADQuery script, you must store the completed CADQuery solid in the `result` variable.
- Use all available perspective renders together if provided.
- Create and run temporary python scripts when you need exact geometry or numerical analysis.
""".strip()
    + "\n"
)


@dataclass(frozen=True)
class MessageBuilder:
    access_render3d: Literal["none", "path", "image"]
    access_render3d_styles: tuple[str, ...]
    feedback_render3d: Literal["none", "path", "image"]
    feedback_render3d_styles: tuple[str, ...]
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

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

    def build_initial(self, manifest: InputManifest) -> list[BaseMessage]:
        blocks: list[ContentBlock] = [
            create_text_block(
                "\n".join(
                    [
                        f"[Input DXF path: {manifest.dxf_path}]",
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

        self._validate_requested_styles(self.access_render3d_styles, manifest)
        self._validate_requested_styles(self.feedback_render3d_styles, manifest)

        blocks.extend(
            self._render_blocks(
                mode=self.access_render3d,
                render3d_styles=self.access_render3d_styles,
                render3d_paths=manifest.render3d_paths,
                label="[Input perspective renders]",
            )
        )

        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content_blocks=blocks),
        ]

    def build_feedback(self, manifest: FeedbackManifest) -> HumanMessage:
        blocks: list[ContentBlock] = [
            create_text_block(
                f"[Candidate execution feedback]\n{manifest.execution_feedback}\n"
            )
        ]

        if manifest.dxf_path is not None:
            blocks.append(
                create_text_block(f"[Projected DXF path: {manifest.dxf_path}]\n")
            )

        blocks.extend(
            self._render_blocks(
                mode=self.feedback_render3d,
                render3d_styles=self.feedback_render3d_styles,
                render3d_paths=manifest.render3d_paths,
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
        render3d_paths: Mapping[str, Path],
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
                blocks.append(create_text_block(f"- {style}"))
                blocks.append(
                    create_image_block(
                        base64=base64.b64encode(
                            render3d_paths[style].read_bytes()
                        ).decode("ascii"),
                        mime_type="image/png",
                    )
                )
            return blocks

        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
