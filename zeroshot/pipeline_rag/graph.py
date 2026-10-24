"""Annotate a drawing for the region index: features described by how views draw them."""

import re
from dataclasses import replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, NotRequired, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.messages.content import ContentBlock, create_text_block
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import Field

from zeroshot.pipeline.messages.artifact import ArtifactPresenter, drawing_for_model
from zeroshot.pipeline.messages.manifest import InputManifest
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages._base.contracts import Submission
from zeroshot.pipeline.stages._base.prompt import (
    PromptTemplate,
    StageInstructions,
    build_system_prompt,
    describe_placement,
    schema_for_prompt,
)
from zeroshot.pipeline.stages.interpretation.contracts import (
    PICTORIAL_VIEWS,
    UNDECIDED,
    DrawingInterpretation,
)
from zeroshot.pipeline.stages.interpretation.verify import (
    InterpretationVerificationResult,
    InterpretationVerifier,
)
from zeroshot.pipeline.tools.fit_edge import create_fit_edge_tool
from zeroshot.pipeline.tools.load_image import create_load_image_tool
from zeroshot.pipeline.tools.run_shell import create_run_shell_tool
from zeroshot.pipeline.verification import AttemptStore
from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.middleware import VerifyOnWriteMiddleware

_PROMPTS = Path(__file__).parent / "prompts"
_FRAMES = (
    Path(__file__).parents[1] / "pipeline/stages/_base/prompts/coordinate_frames.md"
)


class AnnotationReport(Submission):
    """What you are unsure of in the interpretation file you leave."""

    concerns: dict[str, str] = Field(
        ...,
        description=(
            "One concern_... entry per feature or reading you are unsure of, {} if "
            "none. Each value is one string naming the sem_ members, the "
            "readings the views allow and why you chose yours."
        ),
    )


# Numbers (but not 2D/3D), member names, model axes and planes, and view names.
_NOT_CARRIED_OVER = re.compile(
    r"\d+(?:\.\d+)?(?![dD]\b)"
    r"|\b(?:sem|dim|view|concern)_\w+"
    r"|(?<!\w)[+-]?[xyz](?:[- ]?axis)?(?![\w-])"
    r"|\b(?:xy|xz|yz)\b"
    r"|\b(?:front|back|rear|top|bottom|left|right|plan|side)[- ]"
    r"(?:view|projection|elevation)s?\b",
    re.IGNORECASE,
)


def wording_errors(interpretation: DrawingInterpretation) -> list[str]:
    """Descriptions holding words that do not carry over to another part."""
    return [
        f"$.features[{index}].description ({feature.name}) holds "
        f"{', '.join(sorted(set(found)))}. A description carries no numbers, "
        "model axes, view names or member names; name a view by its direction "
        "to the feature."
        for index, feature in enumerate(interpretation.features)
        if (found := _NOT_CARRIED_OVER.findall(feature.description))
    ]


class AnnotationVerifier(InterpretationVerifier):
    """Refuses descriptions that would not carry over to another part, and
    drops the reminder to measure: an annotation carries no sizes."""

    def verify(self) -> InterpretationVerificationResult:
        result = super().verify()
        if (
            result.interpretation is not None
            and not result.errors
            and (errors := wording_errors(result.interpretation))
        ):
            self._built = replace(result, errors=tuple(errors))
        return self._built or result

    def feedback(self) -> list[ContentBlock]:
        report, _measure = super().feedback()
        return [report]


class AnnotationState(TypedDict):
    transcript: NotRequired[list[Any]]
    concerns: NotRequired[dict[str, str] | None]


def create_annotation_graph(
    agent_builder: partial[Any],
    sandbox_runner: SandboxRunner,
    sandbox_workdir: SandboxWorkdir,
    artifact_presenter: ArtifactPresenter,
    input_manifest: InputManifest,
    output_filename: str = "model.py",
    verification_dirname: PurePosixPath = PurePosixPath("attempts"),
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    interpretation_filename: str = "interpretation.json",
):
    """One agent writes the interpretation until it validates, then answers."""
    del output_filename  # nothing is built
    verifier = AnnotationVerifier(
        workdir=sandbox_workdir,
        attempt_store=AttemptStore(
            sandbox_workdir, round_source=lambda: 0, root_dirname=verification_dirname
        ),
        input_artifact=input_manifest.drawing,
        projection_method=input_manifest.projection_method,
        source_filename=interpretation_filename,
    )
    middleware = VerifyOnWriteMiddleware(verifier, fingerprint=verifier.source_digest)
    context = {
        "interpretation_output_path": str(
            sandbox_workdir.sandbox_bind_dir / interpretation_filename
        ),
        "interpretation_schema": schema_for_prompt(DrawingInterpretation),
        "projection_placement": describe_placement(input_manifest.projection_method),
        "pictorial": (
            " " + PromptTemplate(_PROMPTS / "pictorial.md").render()
            if any(view.role in PICTORIAL_VIEWS for view in input_manifest.drawing)
            else ""
        ),
    }
    agent = agent_builder(
        tools=[
            create_run_shell_tool(sandbox_runner, sandbox_workdir),
            create_load_image_tool(sandbox_workdir),
            create_fit_edge_tool(sandbox_workdir),
        ],
        system_prompt=build_system_prompt(_PROMPTS / "role.md", {}),
        output_schema=AnnotationReport,
        extra_middleware=[middleware],
    )
    inputs = StageInstructions(
        input_artifact=input_manifest.drawing,
        input_presentation_mode=artifact_presenter.input,
        prompt_context={},
        workdir=sandbox_workdir,
        projection_method=input_manifest.projection_method,
    ).create_artifact_message()

    def annotate(state: AnnotationState, config: RunnableConfig) -> AnnotationState:
        del state
        verifier.reset(
            DrawingInterpretation(
                datum=UNDECIDED,
                views=drawing_for_model(input_manifest.drawing, sandbox_workdir),
                features=[],
            )
        )
        middleware.reset()
        task = "\n\n".join(
            PromptTemplate(path).render(**context)
            for path in (_FRAMES, _PROMPTS / "task.md")
        )
        instruction = HumanMessage(
            content_blocks=[create_text_block(task), *inputs.content_blocks]
        )
        result = agent.invoke(
            {"messages": [instruction]}, config=_child_graph_config(config)
        )
        answer = result.get("structured_response")
        return {
            "transcript": result["messages"],
            "concerns": answer.concerns
            if isinstance(answer, AnnotationReport)
            else None,
        }

    workflow = StateGraph(AnnotationState)
    workflow.add_node("annotation", annotate)
    workflow.add_edge(START, "annotation")
    workflow.add_edge("annotation", END)
    return workflow.compile(checkpointer=checkpointer)
