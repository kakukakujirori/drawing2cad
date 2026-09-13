"""The C4 graph with one transcript shared by its reasoning stages."""

import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from tests.zeroshot.chat_models import ScriptedChatModel
from tests.zeroshot.prompt_paths import ROLE_PATHS
from tests.zeroshot.workflow.test_graph import (
    _accepted_audit,
    _coding_submission,
    _interpretation_script,
    _interpretation_submission,
    _invalid_interpretation_submission,
    _operation_submission,
    _stub_verification,
    _verified,
    _write_interpretation,
)
from zeroshot.pipeline.messages.artifact import ArtifactPresenter
from zeroshot.pipeline.messages.manifest import InputManifest
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.stages.drawings.contracts import (
    DrawingSource,
    View,
    unread_sheet,
)
from zeroshot.pipeline.stages.types import REASONING_STAGES, PipelineStage
from zeroshot.pipeline.workflow import create_agent
from zeroshot.pipeline.workflow.components.compact import (
    COMPACTION_INSTRUCTION,
    SUMMARY_PREAMBLE,
)
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph

_ROLE = "cad_reconstructor"
_INPUT_MARKER = "[Input artifacts]"


class _Models(TypedDict):
    interpreter: ScriptedChatModel
    planner: ScriptedChatModel
    coder: ScriptedChatModel
    auditor: ScriptedChatModel


def _continued_graph(
    workdir: SandboxWorkdir,
    *,
    interpreter: ScriptedChatModel,
    planner: ScriptedChatModel,
    coder: ScriptedChatModel,
    auditor: ScriptedChatModel,
    share_thread: bool = True,
    **overrides: Any,
):
    common = {
        "announce_turns": False,
        "model_retries": 0,
        "checkpointer": False,
    }
    from PIL import Image

    image_path = workdir.host_bind_dir / "drawing.png"
    Image.new("RGB", (20, 20), "white").save(image_path)
    Image.new("RGB", (20, 20), "white").save(workdir.host_bind_dir / "front.png")
    return create_reconstruction_graph(
        interpretation_agent_builder=partial(
            create_agent,
            role=_ROLE,
            model=interpreter,
            max_turns=5,
            **common,
        ),
        operations_agent_builder=partial(
            create_agent,
            role=_ROLE,
            model=planner,
            max_turns=5,
            **common,
        ),
        coding_agent_builder=partial(
            create_agent,
            role=_ROLE,
            model=coder,
            max_turns=5,
            **common,
        ),
        audit_agent_builder=partial(
            create_agent,
            role="output_auditor",
            model=auditor,
            max_turns=5,
            **common,
        ),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable),
            default_timeout_s=10,
        ),
        sandbox_workdir=workdir,
        artifact_presenter=ArtifactPresenter(input_mode="path", feedback_mode="none"),
        input_manifest=InputManifest(
            sample_id="test",
            drawing=DrawingSource(
                sheets=[unread_sheet("sheet_drawing", View.FULL_PAGE, image_path)]
            ),
        ),
        share_thread=share_thread,
        **overrides,
    )


@pytest.fixture
def models() -> _Models:
    return {
        "interpreter": ScriptedChatModel(responses=_interpretation_script()),
        "planner": ScriptedChatModel(responses=(_operation_submission(),)),
        "coder": ScriptedChatModel(responses=(_coding_submission(),)),
        "auditor": ScriptedChatModel(responses=(_accepted_audit(),)),
    }


def _texts(messages: Sequence[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def _system_prompt(model: ScriptedChatModel) -> str:
    message = model.received_messages[0][0]
    assert isinstance(message, SystemMessage)
    return message.text


def _lead_thread(result: dict[str, Any], stage: PipelineStage) -> list[BaseMessage]:
    if stage is PipelineStage.INTERPRETATION:
        return list(result["interpretation_state"]["messages"])
    if stage is PipelineStage.OPERATIONS:
        return list(result["operations_state"]["messages"])
    return list(result["coding_state"]["messages"])


def test_reasoning_stages_share_one_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    models: _Models,
) -> None:
    _stub_verification(monkeypatch, _verified())
    with SandboxWorkdir() as workdir:
        _continued_graph(workdir, **models).invoke({})

    prompts = {
        _system_prompt(models[name]) for name in ("interpreter", "planner", "coder")
    }
    assert len(prompts) == 1
    (shared_prompt,) = prompts
    assert ROLE_PATHS["cad_reconstructor"].read_text().strip() in shared_prompt
    assert _system_prompt(models["auditor"]) != shared_prompt


def test_handover_nodes_follow_successful_integration(
    models: _Models,
) -> None:
    with SandboxWorkdir() as workdir:
        shared = _continued_graph(workdir, **models).get_graph()

    for stage in REASONING_STAGES:
        handover = f"{stage.value}_handover"
        assert handover in shared.nodes
    assert "integrate_stage_submission" in shared.nodes


def test_each_reasoning_stage_continues_the_preceding_transcript(
    monkeypatch: pytest.MonkeyPatch,
    models: _Models,
) -> None:
    _stub_verification(monkeypatch, _verified())
    with SandboxWorkdir() as workdir:
        _continued_graph(workdir, **models).invoke({})

    interpretation_ask = models["interpreter"].received_messages[-1]
    operation_ask = models["planner"].received_messages[0]
    coding_ask = models["coder"].received_messages[0]
    interpretation_text = "\n".join(_texts(interpretation_ask))
    operation_text = "\n".join(_texts(operation_ask))
    coding_text = "\n".join(_texts(coding_ask))

    assert interpretation_text in operation_text
    assert operation_text in coding_text


def test_shared_thread_retries_interpretation_before_handing_over(
    monkeypatch: pytest.MonkeyPatch,
    models: _Models,
) -> None:
    _stub_verification(monkeypatch, _verified())
    interpreter = ScriptedChatModel(
        responses=(
            _write_interpretation(),
            _invalid_interpretation_submission(),
            _interpretation_submission(),
        )
    )
    models["interpreter"] = interpreter

    with SandboxWorkdir() as workdir:
        result = _continued_graph(
            workdir,
            max_stage_validation_retries=1,
            **models,
        ).invoke({})

    assert len(interpreter.received_messages) == 3
    retry_text = "\n".join(_texts(interpreter.received_messages[2]))
    assert "Interpretation Validation Error" in retry_text
    assert "ticket_absent" in retry_text
    operation_text = "\n".join(_texts(models["planner"].received_messages[0]))
    assert "Interpretation Validation Error" in operation_text
    assert (
        result["reconstruction"].snapshots[0].last_completed_stage
        is PipelineStage.CODING
    )


def test_reasoning_states_end_with_the_same_latest_thread(
    monkeypatch: pytest.MonkeyPatch,
    models: _Models,
) -> None:
    _stub_verification(monkeypatch, _verified())
    with SandboxWorkdir() as workdir:
        result = _continued_graph(workdir, **models).invoke({})

    threads = [_texts(_lead_thread(result, stage)) for stage in REASONING_STAGES]
    assert threads[0] == threads[1] == threads[2]
    # The answer itself, as its own JSON rather than wrapped in a tool call.
    assert any(
        "CodingSubmission" not in text and '"stage":"coding"' in text
        for text in threads[0]
    )


def test_the_auditor_remains_outside_the_shared_thread(
    monkeypatch: pytest.MonkeyPatch,
    models: _Models,
) -> None:
    _stub_verification(monkeypatch, _verified())
    with SandboxWorkdir() as workdir:
        _continued_graph(workdir, **models).invoke({})

    audit_prompt = _texts(models["auditor"].received_messages[0])
    assert sum(_INPUT_MARKER in text for text in audit_prompt) == 1
    assert not any('"deliverable"' in text for text in audit_prompt)


def _notetaker(notes: str = "measured the source views") -> ScriptedChatModel:
    return ScriptedChatModel(
        responses=tuple(AIMessage(content=notes) for _ in REASONING_STAGES)
    )


def test_compaction_hands_the_next_stage_notes_instead_of_full_turns(
    monkeypatch: pytest.MonkeyPatch,
    models: _Models,
) -> None:
    _stub_verification(monkeypatch, _verified())
    notetaker = _notetaker()
    with SandboxWorkdir() as workdir:
        _continued_graph(
            workdir,
            compact_between_stages=notetaker,
            **models,
        ).invoke({})

    planner_prompt = _texts(models["planner"].received_messages[0])
    assert SUMMARY_PREAMBLE.format(notes="measured the source views") in planner_prompt
    assert len(notetaker.received_messages) == len(REASONING_STAGES)
    assert all(
        asked[-1].text == COMPACTION_INSTRUCTION
        for asked in notetaker.received_messages
    )


def test_compaction_without_a_shared_thread_is_refused(
    models: _Models,
) -> None:
    with (
        SandboxWorkdir() as workdir,
        pytest.raises(ValueError, match="needs share_thread"),
    ):
        _continued_graph(
            workdir,
            share_thread=False,
            compact_between_stages=_notetaker(),
            **models,
        )
