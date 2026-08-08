"""One artifact, proposed by one role and accepted by another.

The shape every reasoning stage of this workflow has: someone proposes, someone
else judges, and the proposal comes back with feedback until it is accepted or
the workflow runs out of patience.  What differs between stages is the contract
the proposer owes and the words used to ask for it -- which is all
`ProposerCriticSpec` holds -- so a second stage is a declaration rather than a
second copy of this loop.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from zeroshot.pipeline.messages import build_instruction
from zeroshot.pipeline.workflow.agent import AgentFactory, run_agent
from zeroshot.pipeline.workflow.state import Review, StopReason

StructuredOutput = Literal["provider", "tool"]

_STRATEGIES = {"provider": ProviderStrategy, "tool": ToolStrategy}


@dataclass(frozen=True)
class ProposerCriticSpec:
    """What a stage's proposer owes, and where the words to ask for it live.

    Code rather than config: that a reviewer of hypotheses returns a verdict on
    a hypothesis is not an experiment anyone runs.  Which model answers for a
    role, and how long it may take, is config.

    `instructions` names a directory under `prompts/instructions/` holding
    `propose.md`, `revise.md` and `review.md`, so one stage's wording sits
    together and can be edited without touching Python.
    """

    proposal: type[BaseModel]
    instructions: str


@dataclass(frozen=True)
class StageResult:
    """What a stage hands back to the workflow.

    `turns` is what the stage cost, over every agent and every round; the stop
    reason is the one that ended it.
    """

    artifact: BaseModel | None
    messages: list[BaseMessage]
    turns: int
    stop_reason: StopReason | None


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
    strategy = _STRATEGIES[structured_output]

    def build_agent(agent_factory: AgentFactory, schema: type[BaseModel]):
        # The schema reaches the model twice from one definition: as the
        # provider's own output contract, and as the `$output_schema` its
        # prompt explains.  A role prompt that spelled it out by hand would be
        # a second definition to keep in step.
        return agent_factory(
            tools=list(tools),
            prompt_context={
                **prompt_context,
                "output_schema": json.dumps(schema.model_json_schema(), indent=2),
            },
            response_format=strategy(schema=schema),
            model_retries=model_retries,
        )

    agent_proposer = build_agent(proposer, spec.proposal)
    agent_critic = build_agent(critic, Review)

    def instruct(name: str, **context: str):
        return build_instruction(f"{spec.instructions}/{name}", **context)

    def run(history: Sequence[BaseMessage], **upstream: str) -> StageResult:
        """Propose and review until accepted, or until patience runs out.

        Each role reads the run so far plus its own turns in this stage, and
        not the other's.  A critic that had read how the proposal was arrived at
        would be judging the reasoning it just agreed with; whatever it does
        need -- the drawing, the artifact, what an earlier stage settled -- it
        is told in its instruction.
        """
        proposer_view: list[BaseMessage] = []
        critic_view: list[BaseMessage] = []
        delta: list[BaseMessage] = []
        artifact: BaseModel | None = None
        feedback: str | None = None
        turns = 0
        stop_reason: StopReason | None = None

        for _ in range(max_revisions + 1):
            proposed = run_agent(
                agent_proposer,
                [*history, *proposer_view],
                instruct("propose", **upstream)
                if feedback is None
                else instruct("revise", feedback=feedback, **upstream),
                spec.proposal,
            )
            proposer_view += proposed.messages
            delta += proposed.messages
            turns += proposed.turns
            stop_reason = proposed.stop_reason
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
            turns += reviewed.turns
            stop_reason = reviewed.stop_reason
            if reviewed.answer is None or reviewed.answer.decision == "accept":
                break
            feedback = reviewed.answer.feedback

        return StageResult(artifact, delta, turns, stop_reason)

    return run
