"""Proposer-Reviewer bouncing graph template."""

from collections.abc import Mapping, Sequence
from inspect import cleandoc
from types import MappingProxyType
from typing import Literal, NotRequired, Self, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.types import Checkpointer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.agent import AgentState, create_agent
from zeroshot.pipeline.workflow.middleware import StopReason


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal: list[str] = Field(
        ...,
        description="List of proposals.",
    )
    rationale: str = Field(
        ...,
        description="Rationale for the proposal.",
    )


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accept: bool = Field(
        ...,
        description="True if the proposal is acceptable, False if it needs revision.",
    )
    rationale: str = Field(
        ...,
        description="Why that decision. If not accept, name what is wrong and what it should be instead, specific enough to act on without guessing; it must not be empty. If accept, summarize what makes the proposal correct.",
    )

    @model_validator(mode="after")
    def require_rationale_for_revision(self) -> Self:
        if not self.accept and not self.rationale.strip():
            raise ValueError("rationale is required when accept is False")
        return self


class ProposerReviewerState(TypedDict):
    """States that are specific to the proposer-reviewer subgraph.

    States that are specific to the proposer or reviewer are also managed here.
    """

    # general
    revision_count: NotRequired[int]

    # proposer
    proposer_entry_instruction: HumanMessage
    proposal: NotRequired[Proposal | None]
    proposer_state: NotRequired[AgentState | None]

    # reviewer
    reviewer_entry_instruction: HumanMessage
    review: NotRequired[Review | None]
    reviewer_state: NotRequired[AgentState | None]


def create_proposer_reviewer_loop(
    proposer_role: str,
    proposer_model: BaseChatModel,
    reviewer_role: str,
    reviewer_model: BaseChatModel,
    tools: Sequence[BaseTool],
    prompt_context: Mapping[str, str] = MappingProxyType({}),
    response_format_strategy: Literal["provider", "tool"] = "provider",
    max_proposer_turns_per_revision: int = 30,
    max_reviewer_turns_per_revision: int = 30,
    announce_turns: bool = True,
    reset_proposer_turns_per_revision: bool = True,
    reset_reviewer_turns_per_revision: bool = True,
    max_revisions: int = 30,
    announce_revisions: bool = False,
    reset_revision_count_when_reentrant: bool = True,
    model_retries: int = 5,
    checkpointer: Checkpointer = False,
):
    """Build a proposer-reviewer loop that stops on acceptance or its limits."""
    if max_revisions < 0:
        raise ValueError(f"{max_revisions=} must be non-negative")

    # TODO: IMPLEMENT
    if announce_revisions:
        raise NotImplementedError(f"{announce_revisions=} is not implemented yet.")

    def agent_builder(
        role: str,
        model: BaseChatModel,
        schema: type[BaseModel],
        max_turns: int,
    ):
        return create_agent(
            role=role,
            model=model,
            tools=list(tools),
            prompt_context=prompt_context,
            output_schema=schema,
            response_format_strategy=response_format_strategy,
            max_turns=max_turns,
            announce_turns=announce_turns,
            reset_turns_when_reentrant=False,
            model_retries=model_retries,
            checkpointer=False,  # AgentState is managed by ProposerReviewerState
        )

    proposer = agent_builder(
        proposer_role,
        proposer_model,
        Proposal,
        max_proposer_turns_per_revision,
    )
    reviewer = agent_builder(
        reviewer_role,
        reviewer_model,
        Review,
        max_reviewer_turns_per_revision,
    )

    def initialize(state: ProposerReviewerState):

        # sanity check
        if (
            not reset_revision_count_when_reentrant
            and state.get("revision_count", 0) >= max_revisions > 0
        ):
            raise ValueError(f"Revision count reached max revisions: {max_revisions}")

        # append instruction messages to proposer_state
        past_proposer_state = state.get("proposer_state") or {}
        past_proposer_messages = past_proposer_state.get("messages") or []
        proposer_state = {
            **past_proposer_state,
            "messages": [
                *past_proposer_messages,
                state["proposer_entry_instruction"],
            ],
            "current_turn": 0,
        }

        # append instruction messages to reviewer_state
        past_reviewer_state = state.get("reviewer_state") or {}
        past_reviewer_messages = past_reviewer_state.get("messages") or []
        reviewer_state = {
            **past_reviewer_state,
            "messages": [
                *past_reviewer_messages,
                state["reviewer_entry_instruction"],
            ],
            "current_turn": 0,
        }

        update = {
            "proposal": None,
            "proposer_state": proposer_state,
            "review": None,
            "reviewer_state": reviewer_state,
        }

        if reset_revision_count_when_reentrant:
            update |= {"revision_count": 0}

        return update

    def propose(state: ProposerReviewerState, config: RunnableConfig) -> dict:
        """Ask for a proposal: the workflow's question, or the reviewer's."""
        previous = state.get("proposer_state") or {}
        messages = list(previous.get("messages") or [])  # avoid overwrite

        if review := state.get("review"):
            review_message = cleandoc(f"""
                [Revision Request]
                Revise the previous proposal according to this review:

                {review.rationale}
                """)
            messages.append(HumanMessage(content=review_message))

        result = proposer.invoke(
            {**previous, "messages": messages},
            config=_child_graph_config(config),
        )

        update = {
            "proposal": result.get("structured_response") or state.get("proposal"),
            "proposer_state": result,
            "review": None,  # since it was used above
        }

        # reset reviewer counter
        if reset_reviewer_turns_per_revision:  # noqa: SIM102
            # reset only when proposal is valid
            if update["proposal"]:
                reviewer_state = state.get("reviewer_state") or {}
                update["reviewer_state"] = {
                    **reviewer_state,
                    "current_turn": 0,
                }

        return update

    def after_propose(
        state: ProposerReviewerState,
    ) -> Literal["review", "propose", "__end__"]:
        """Decide whether to review or regenerate."""
        if state.get("proposal"):
            # A stage given no review budget answers with its first proposal.
            if state.get("revision_count", 0) >= max_revisions:
                return "__end__"
            return "review"
        elif proposer_state := state.get("proposer_state"):
            if proposer_state.get("stop_reason") == StopReason.BUDGET_EXHAUSTED:
                return "__end__"
            else:
                # Internal error, try again
                return "propose"
        else:
            # Abnormal case, try again
            return "propose"

    def review(state: ProposerReviewerState, config: RunnableConfig) -> dict:
        """Ask the reviewer to review a proposal."""
        previous = state.get("reviewer_state") or {}
        messages = list(previous.get("messages") or [])  # avoid overwrite

        if proposal := state.get("proposal"):
            proposal_list = "\n".join(f"- {item}" for item in proposal.proposal)
            proposal_message = cleandoc(f"""
                [Proposal]
                {proposal_list}

                [Rationale]
                {proposal.rationale}
                """)
            messages.append(HumanMessage(content=proposal_message))
        result = reviewer.invoke(
            {**previous, "messages": messages},
            config=_child_graph_config(config),
        )

        # increment revision_count
        update = {
            "review": result.get("structured_response"),
            "reviewer_state": result,
        }
        if update["review"]:
            update |= {"revision_count": state.get("revision_count", 0) + 1}

        # reset proposer counter
        if reset_proposer_turns_per_revision:
            review_result = result.get("structured_response")
            # reset only when review is required
            if review_result and not review_result.accept:
                proposer_state = state.get("proposer_state") or {}
                update["proposer_state"] = {
                    **proposer_state,
                    "current_turn": 0,
                }

        return update

    def after_review(
        state: ProposerReviewerState,
    ) -> Literal["propose", "review", "__end__"]:
        """Decide whether to accept or revise."""
        review = state.get("review")
        if review:
            revision_count = state.get("revision_count", 0)  # NOTE: already incremented
            if review.accept or revision_count >= max_revisions:
                return "__end__"
            else:
                return "propose"
        elif review_state := state.get("reviewer_state"):
            if review_state.get("stop_reason") == StopReason.BUDGET_EXHAUSTED:
                return "__end__"
            else:
                # Internal error, try again
                return "review"
        else:
            # Abnormal case, try again
            return "review"

    stage = StateGraph(state_schema=ProposerReviewerState)  # type: ignore[type-var]

    stage.add_node("initialize", initialize)
    stage.add_node("propose", propose)
    stage.add_node("review", review)

    stage.add_edge(START, "initialize")
    stage.add_edge("initialize", "propose")
    stage.add_conditional_edges("propose", after_propose)
    stage.add_conditional_edges("review", after_review)
    graph = stage.compile(checkpointer=checkpointer)
    # initialize is one superstep and each complete revision is one propose +
    # one review.  Leave headroom for the terminal routing superstep.
    return graph.with_config(recursion_limit=2 * max_revisions + 4)
