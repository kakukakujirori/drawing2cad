"""One artifact, proposed by one role and accepted by another.

The shape every reasoning stage of this workflow has: someone proposes, someone
else judges, and the proposal comes back with feedback until it is accepted or
the workflow runs out of patience.  What differs between stages is the contract
the proposer owes and the words used to ask for it -- which is all
`ProposerCriticSpec` holds -- so a second stage is a declaration rather than a
second copy of this loop.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from zeroshot.pipeline.messages import build_instruction
from zeroshot.pipeline.workflow.agent import (
    AgentFactory,
    StructuredOutput,
    run_agent,
)
from zeroshot.pipeline.workflow.state import Review, StopReason, add_turns

# Why a node is running, which is also the name of the instruction it opens on.
# A stage is asked for the first time, asked again for a fault the audit found
# in its own answer, or asked again because what it worked from has changed.
Opening = Literal["propose", "revise", "restate"]


@dataclass(frozen=True)
class ProposerCriticSpec:
    """What a stage's proposer owes, and where the words to ask for it live.

    Code rather than config: that a reviewer of hypotheses returns a verdict on
    a hypothesis is not an experiment anyone runs.  Which model answers for a
    role, and how long it may take, is config.

    `instructions` names a directory under `prompts/instructions/` holding one
    file per way the stage can be asked -- `review.md` for the critic, and one
    per `opening` the workflow passes -- so a stage's wording sits together and
    can be edited without touching Python.
    """

    proposal: type[BaseModel]
    instructions: str


@dataclass(frozen=True)
class StageResult:
    """What a stage hands back to the workflow.

    Both tallies are keyed by role: a proposer that ran out of turns and a
    critic that answered are two facts, and one of them is why the stage has no
    artifact.  `turns` covers every round the stage took.
    """

    artifact: BaseModel | None
    messages: list[BaseMessage]
    turns: dict[str, int]
    stop_reasons: dict[str, StopReason]


StageRunner = Callable[..., StageResult]
StageFactory = Callable[..., StageRunner]


def create_proposer_critic_loop(
    spec: ProposerCriticSpec,
    proposer: AgentFactory,
    critic: AgentFactory,
    *,
    max_revisions: int,
    structured_output: StructuredOutput,
    tools: Sequence[BaseTool],
    prompt_context: Mapping[str, str],
    model_retries: int = 5,
) -> StageRunner:
    """Build the stage: two agents, and the loop that makes them agree."""
    if max_revisions < 0:
        raise ValueError("max_revisions must not be negative")

    def build_agent(agent_factory: AgentFactory, schema: type[BaseModel]):
        return agent_factory(
            tools=list(tools),
            prompt_context=prompt_context,
            output_schema=schema,
            structured_output=structured_output,
            model_retries=model_retries,
        )

    agent_proposer = build_agent(proposer, spec.proposal)
    agent_critic = build_agent(critic, Review)

    def instruct(name: str, **context: str):
        return build_instruction(f"{spec.instructions}/{name}", **context)

    def run(
        history: Sequence[BaseMessage],
        *,
        opening: Opening = "propose",
        feedback: str | None = None,
        **upstream: str,
    ) -> StageResult:
        """Propose and review until accepted, or until patience runs out.

        Each role reads the run so far plus its own turns in this stage, and
        not the other's.  A critic that had read how the proposal was arrived at
        would be judging the reasoning it just agreed with; whatever it does
        need -- the drawing, the artifact, what an earlier stage settled -- it
        is told in its instruction.

        `opening` names the instruction the stage starts on, which is how the
        workflow says why it is running at all.  After the first round it is
        always a revision, because from there it is the critic asking.
        """
        proposer_view: list[BaseMessage] = []
        critic_view: list[BaseMessage] = []
        delta: list[BaseMessage] = []
        artifact: BaseModel | None = None
        turns: dict[str, int] = {}
        stop_reasons: dict[str, StopReason] = {}
        asked: Opening = opening
        context = {} if feedback is None else {"feedback": feedback}

        for _ in range(max_revisions + 1):
            proposed = run_agent(
                agent_proposer,
                [*history, *proposer_view],
                instruct(asked, **context, **upstream),
                spec.proposal,
            )
            proposer_view += proposed.messages
            delta += proposed.messages
            turns = add_turns(turns, proposed.turns_by_role)
            stop_reasons |= proposed.stop_reasons
            if proposed.answer is None:
                break
            artifact = proposed.answer

            reviewed = run_agent(
                agent_critic,
                [*history, *critic_view],
                instruct(
                    "review", proposal=artifact.model_dump_json(indent=2), **upstream
                ),
                Review,
            )
            critic_view += reviewed.messages
            delta += reviewed.messages
            turns = add_turns(turns, reviewed.turns_by_role)
            stop_reasons |= reviewed.stop_reasons
            if reviewed.answer is None or reviewed.answer.decision == "accept":
                break
            asked, context = "revise", {"feedback": reviewed.answer.feedback}

        return StageResult(artifact, delta, turns, stop_reasons)

    return run
