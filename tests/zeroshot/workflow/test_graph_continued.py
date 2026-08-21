"""The staged graph run as one agent: one model, one prompt, one transcript."""

import json
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from tests.zeroshot.chat_models import ScriptedChatModel
from zeroshot.pipeline.messages import ArtifactPresenter, InputManifest
from zeroshot.pipeline.sandbox import SandboxRunner, SandboxWorkdir
from zeroshot.pipeline.verification import StepRenderer
from zeroshot.pipeline.workflow import (
    FanoutReduceProposal,
    Proposal,
    create_agent,
    create_fanout_reduce_graph,
    create_proposer_reviewer_loop,
)
from zeroshot.pipeline.workflow.components.compact import (
    COMPACTION_INSTRUCTION,
    SUMMARY_PREAMBLE,
)
from zeroshot.pipeline.workflow.graph import create_reconstruction_graph

ROLE = "cad_reconstructor"
INPUT_MARKER = "[Input DXF path:"
_REASONING_STAGES = ("semantics", "operations", "coding")


def _hypothesis(*items: str) -> AIMessage:
    return AIMessage(
        content=FanoutReduceProposal(
            proposal=list(items), rationale="the views agree"
        ).model_dump_json()
    )


def _plan(*items: str) -> AIMessage:
    return AIMessage(
        content=Proposal(
            proposal=list(items), rationale="the views agree"
        ).model_dump_json()
    )


def _audit(revise: str | None, rationale: str = "") -> AIMessage:
    return AIMessage(content=json.dumps({"revise": revise, "rationale": rationale}))


def _continued_graph(
    workdir: SandboxWorkdir,
    *,
    lead: ScriptedChatModel,
    other_proposer: ScriptedChatModel,
    planner: ScriptedChatModel,
    coder: ScriptedChatModel,
    auditor: ScriptedChatModel | None = None,
    reviewer: ScriptedChatModel | None = None,
    share_thread: bool = True,
    **overrides: Any,
):
    """What `configs/workflow/continued.yaml` builds, with scripted models.

    The three reasoning stages take one role, as they must for a shared
    transcript; the fan-out's other proposer and the audit keep their own.
    """
    common = {"announce_turns": False, "model_retries": 5, "checkpointer": False}
    dxf_path = workdir.host_bind_dir / "drawing.dxf"
    dxf_path.write_text("0\nSECTION\n0\nEOF\n", encoding="utf-8")
    return create_reconstruction_graph(
        semantics_agent_builder=partial(
            create_fanout_reduce_graph,
            proposer_role=ROLE,
            proposer_models=[lead, other_proposer],
            max_proposer_turns=5,
            max_reducer_turns=5,
            **common,
        ),
        operations_agent_builder=partial(
            create_proposer_reviewer_loop,
            proposer_role=ROLE,
            proposer_model=planner,
            reviewer_role="operation_reviewer",
            reviewer_model=reviewer or ScriptedChatModel(responses=()),
            max_revisions=0,  # as shipped: the audit after coding is the review
            max_proposer_turns_per_revision=5,
            max_reviewer_turns_per_revision=5,
            **common,
        ),
        coding_agent_builder=partial(
            create_agent, role=ROLE, model=coder, max_turns=5, **common
        ),
        audit_agent_builder=partial(
            create_agent,
            role="output_auditor",
            model=auditor or ScriptedChatModel(responses=(_audit(None, "matches"),)),
            max_turns=5,
            **common,
        ),
        sandbox_runner=SandboxRunner(
            python_executable=Path(sys.executable), default_timeout_s=10
        ),
        sandbox_workdir=workdir,
        renderer=StepRenderer(timeout_s=60.0),
        artifact_presenter=ArtifactPresenter(
            input_render3d_mode="none",
            input_render3d_styles=(),
            feedback_render3d_mode="none",
            feedback_render3d_styles=(),
        ),
        input_manifest=InputManifest(
            sample_id="test", dxf_path=dxf_path, render3d_paths={}
        ),
        share_thread=share_thread,
        **overrides,
    )


def _run(workdir: SandboxWorkdir, **kwargs: Any) -> dict[str, Any]:
    return _continued_graph(workdir, **kwargs).invoke({})


def _texts(messages: Sequence[BaseMessage]) -> list[str]:
    return [message.text for message in messages]


def _first_ask(model: ScriptedChatModel) -> list[BaseMessage]:
    return model.received_messages[0]


def _last_ask(model: ScriptedChatModel) -> list[BaseMessage]:
    return model.received_messages[-1]


def _system_prompt(model: ScriptedChatModel) -> str:
    first = _first_ask(model)[0]
    assert isinstance(first, SystemMessage)
    return first.text


def _lead_thread(result: dict[str, Any], stage: str) -> list[BaseMessage]:
    if stage == "semantics":
        return list(result["semantics_state"]["reducer_state"]["messages"])
    if stage == "operations":
        return list(result["operations_state"]["proposer_state"]["messages"])
    return list(result["coding_state"]["messages"])


@pytest.fixture
def sandbox_workdir():
    with SandboxWorkdir() as workdir:
        yield workdir


@pytest.fixture
def models() -> dict[str, ScriptedChatModel]:
    return {
        "lead": ScriptedChatModel(
            responses=(_hypothesis("a flanged boss"), _hypothesis("a flanged boss"))
        ),
        "other_proposer": ScriptedChatModel(responses=(_hypothesis("a boss"),)),
        "planner": ScriptedChatModel(responses=(_plan("extrude the outline"),)),
        "coder": ScriptedChatModel(responses=(AIMessage(content="written"),)),
    }


def test_the_three_reasoning_stages_are_given_one_system_prompt(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """A transcript reread under a persona that did not produce it is a
    contradiction, so the thread's stages must share the prompt as well."""
    _run(sandbox_workdir, **models)

    prompts = {
        stage: _system_prompt(models[stage]) for stage in ("lead", "planner", "coder")
    }
    assert len(set(prompts.values())) == 1, prompts
    # And it is the merged role, not one of the single-stage ones.
    assert "Phase 1" in prompts["lead"]


@pytest.mark.parametrize("stage", _REASONING_STAGES)
def test_a_stage_hands_its_thread_over_before_the_run_moves_on(
    sandbox_workdir: SandboxWorkdir,
    models: dict[str, ScriptedChatModel],
    stage: str,
) -> None:
    """The two modes differ by wiring alone: the stages themselves are the same
    nodes either way, and only a shared thread adds one to hand it over."""
    shared = _continued_graph(sandbox_workdir, **models).get_graph()
    staged = _continued_graph(sandbox_workdir, share_thread=False, **models).get_graph()

    handover = f"{stage}_handover"
    assert handover in shared.nodes
    assert handover not in staged.nodes
    # Nothing leaves the stage except the handover, so nothing runs on a thread
    # the stage before it has not left behind.
    assert [edge.target for edge in shared.edges if edge.source == stage] == [handover]
    assert {edge.target for edge in shared.edges if edge.source == handover} == {
        edge.target for edge in staged.edges if edge.source == stage
    }


def test_a_stage_continues_the_transcript_of_the_stage_before_it(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """What each stage is asked, not what the run stored: by the end every
    stage state holds the same thread, so only the asks show it growing."""
    _run(sandbox_workdir, **models)

    semantics = _last_ask(models["lead"])
    operations = _first_ask(models["planner"])
    coding = _first_ask(models["coder"])

    assert _texts(operations)[: len(semantics)] == _texts(semantics)
    assert _texts(coding)[: len(operations)] == _texts(operations)
    assert len(coding) > len(operations) > len(semantics)


def test_every_reasoning_stage_ends_holding_the_newest_thread(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """Each stage is given the thread whether or not it is still to come, so
    that a stage the audit sends back to resumes from where the run got to."""
    result = _run(sandbox_workdir, **models)

    threads = [_texts(_lead_thread(result, stage)) for stage in _REASONING_STAGES]
    assert threads[0] == threads[1] == threads[2]
    assert threads[0] == _texts(_first_ask(models["coder"])[1:]) + ["written"]


def test_each_stage_is_shown_the_drawing_again_when_its_own_work_begins(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """The thread carries what was read out of the drawing, but a stage that
    only inherits the reading never measures against the drawing itself, so
    each opening ask puts it back in front of the model."""
    _run(sandbox_workdir, **models)

    seen = _texts(_first_ask(models["coder"]))
    assert sum(INPUT_MARKER in text for text in seen) == len(_REASONING_STAGES)


def test_the_other_proposer_and_the_audit_read_the_drawing_themselves(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """An independent reading is what they are for, so they stay outside the
    thread and are still shown the inputs."""
    result = _run(sandbox_workdir, **models)

    # The system prompt, the opening instruction carrying the drawing, and its
    # own workspace note -- nothing of what the lead proposer went on to do.
    other_ask = _first_ask(models["other_proposer"])
    assert len(other_ask) == 3
    assert any(INPUT_MARKER in message.text for message in other_ask)
    assert len(models["other_proposer"].received_messages) == 1

    # The audit's own ask and its own answer, opened on the drawing.
    audit = _texts(result["audit_state"]["messages"])
    assert len(audit) == 2
    assert sum(INPUT_MARKER in text for text in audit) == 1


def test_a_stage_the_audit_sends_back_continues_the_same_thread(
    sandbox_workdir: SandboxWorkdir,
) -> None:
    """One engineer carrying on, so the redone stage sees why its plan failed."""
    result = _run(
        sandbox_workdir,
        lead=ScriptedChatModel(
            responses=(_hypothesis("a flanged boss"), _hypothesis("a flanged boss"))
        ),
        other_proposer=ScriptedChatModel(responses=(_hypothesis("a boss"),)),
        planner=ScriptedChatModel(responses=(_plan("extrude"), _plan("revolve"))),
        coder=ScriptedChatModel(
            responses=(AIMessage(content="written"), AIMessage(content="rewritten"))
        ),
        auditor=ScriptedChatModel(
            responses=(
                _audit("operations", "the base is revolved, not extruded"),
                _audit(None, "matches"),
            )
        ),
        max_audit_reject_count=1,
    )

    operations = _texts(_lead_thread(result, "operations"))
    assert "written" in operations
    assert any("the base is revolved" in text for text in operations)


def _notetaker(
    notes: str = "measured 100mm across the front view",
) -> ScriptedChatModel:
    """Stands in for the model whose conversation is being traded for notes."""
    return ScriptedChatModel(
        responses=tuple(AIMessage(content=notes) for _ in range(3))
    )


def test_a_stage_hands_on_notes_instead_of_its_turns_when_asked_to(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """The turns have served their purpose once the stage's answer is settled,
    and the next stage is handed that answer in its own instruction."""
    notetaker = _notetaker()

    _run(sandbox_workdir, compact_between_stages=notetaker, **models)

    planner = _texts(_first_ask(models["planner"]))
    # The system prompt, the notes standing in for semantics, and its own ask.
    assert len(planner) == 3
    assert planner[1] == SUMMARY_PREAMBLE.format(
        notes="measured 100mm across the front view"
    )
    assert "Using the current semantic hypothesis" in planner[2]


def test_the_notes_are_written_by_the_model_that_held_the_conversation(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    notetaker = _notetaker()

    _run(sandbox_workdir, compact_between_stages=notetaker, **models)

    # One handover per reasoning stage, each shown the thread it is reading back.
    assert len(notetaker.received_messages) == len(_REASONING_STAGES)
    for asked in notetaker.received_messages:
        assert asked[-1].text == COMPACTION_INSTRUCTION
    assert any(
        INPUT_MARKER in message.text for message in notetaker.received_messages[0]
    )


def test_the_thread_is_handed_on_whole_when_no_model_is_given(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """Compaction is a choice about cost, so the default keeps every turn."""
    _run(sandbox_workdir, **models)

    assert all("<summary>" not in text for text in _texts(_first_ask(models["coder"])))


def test_compaction_without_a_shared_thread_is_refused(
    sandbox_workdir: SandboxWorkdir, models: dict[str, ScriptedChatModel]
) -> None:
    """With a transcript per stage there is no handover to compact at, so the
    setting would silently do nothing."""
    with pytest.raises(ValueError, match="needs share_thread"):
        _continued_graph(
            sandbox_workdir,
            share_thread=False,
            compact_between_stages=_notetaker(),
            **models,
        )
