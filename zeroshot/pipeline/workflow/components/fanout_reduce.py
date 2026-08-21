"""Fan-out / reduce graph template: independent proposers, one merged answer."""

import operator
import shlex
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Checkpointer
from pydantic import BaseModel, ConfigDict, Field

from zeroshot.pipeline.workflow._config import _child_graph_config
from zeroshot.pipeline.workflow.components.agent import AgentState, create_agent


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


class FanoutReduceState(TypedDict):
    """States that are specific to the fan-out/reduce subgraph.

    Branch proposals are keyed by branch name, and the key carries a reducer
    because every branch writes it within one superstep.  Only the first
    proposer's agent state is retained: it becomes the reducer's state.
    """

    # one instruction per subgraph invocation; cleared after it is consumed
    invocation_instruction: NotRequired[HumanMessage | None]

    # proposers
    branch_proposals: NotRequired[Annotated[dict[str, Proposal | None], operator.or_]]

    # reducer
    proposal: NotRequired[Proposal | None]
    reducer_state: NotRequired[AgentState | None]


def _proposal_lines(header: str, proposal: Proposal) -> list[str]:
    return [
        f"[{header} Proposal]",
        *(f"- {item}" for item in proposal.proposal),
        "",
        f"[{header} Rationale]",
        proposal.rationale,
    ]


def _workspace_message(workdir: str) -> HumanMessage:
    """Tell a proposer where to keep this task's intermediate files."""
    lines = [
        "[Workspace]",
        f"Use {workdir} as your dedicated workspace for this task.",
        "Create or modify all temporary and intermediate files only inside this",
        "directory. You may read shared inputs and other resources from their",
        "provided locations.",
        "",
    ]
    return HumanMessage(content="\n".join(lines))


def _other_proposals_message(proposals: Sequence[Proposal]) -> HumanMessage:
    """Ask the head proposer to reconcile only the other models' answers."""
    lines = [
        "[Reduction Request]",
        "Other models produced the proposals below. Compare them with your own",
        "preceding proposal, if you produced one, and return one final",
        "consolidated proposal:",
        "",
        "- Merge equivalent or duplicate proposals into one item.",
        "- Adopt proposals absent from your answer when they appear correct.",
        "- Resolve conflicting proposals in favor of the conclusion most likely",
        "  to be correct.",
        "- Return the complete final proposal list, not a review or a list of",
        "  changes, and update the rationale to explain the resolved answer.",
        "",
    ]
    for number, proposal in enumerate(proposals, start=1):
        lines.extend(_proposal_lines(f"Other Model {number}", proposal))
        lines.append("")
    return HumanMessage(content="\n".join(lines).strip())


def _current_proposal_message(proposal: Proposal) -> HumanMessage:
    """Make the workflow's adopted answer explicit before a revision."""
    lines = [
        "[Revision Context]",
        "The workflow currently adopts the proposal below as its answer. Treat",
        "it as the proposal to revise, even if the preceding transcript ended",
        "before you returned a final answer.",
        "",
        *_proposal_lines("Current", proposal),
    ]
    return HumanMessage(content="\n".join(lines))


def create_fanout_reduce_graph(
    proposer_role: str,
    proposer_models: Sequence[BaseChatModel],
    tools: Sequence[BaseTool],
    fanout_workdir_prefix: str,
    prompt_context: Mapping[str, str] = MappingProxyType({}),
    response_format_strategy: Literal["provider", "tool"] = "provider",
    max_proposer_turns: int = 30,
    max_reducer_turns: int = 30,
    announce_turns: bool = True,
    model_retries: int = 5,
    checkpointer: Checkpointer = False,
):
    """Answer once with every model, then refine with the head model.

    One proposer is built per entry of `proposer_models`; they never see each
    other's work.  The reducer reuses the first proposer's model, system role,
    and message history, so the merge costs one continuation instead of a
    second reading of the inputs.  On subgraph reentry, the reducer revises its
    prior answer directly instead of running the fan-out again.

    The branches run in one superstep and share `tools`.  Each proposer is told
    to place its writes below a distinct path derived from
    `fanout_workdir_prefix`; the caller owns the path's meaning and location.
    Before the fan-out, the graph creates every path through the one `run_shell`
    tool that `tools` must provide.

    Every invocation must supply a fresh `invocation_instruction`.  The graph
    clears it after a successful pass so a later reentry cannot accidentally
    reuse a stale instruction.
    """
    if not proposer_models or len(proposer_models) <= 1:
        raise ValueError("proposer_models must be >= 2")
    if not fanout_workdir_prefix.strip():
        raise ValueError("fanout_workdir_prefix must not be empty")
    if fanout_workdir_prefix.endswith("/"):
        raise ValueError("fanout_workdir_prefix must not end with '/'")

    run_shell_tools = [tool for tool in tools if tool.name == "run_shell"]
    if len(run_shell_tools) != 1:
        raise ValueError("tools must contain exactly one run_shell tool")
    run_shell = run_shell_tools[0]

    branch_names = [f"propose_{index}" for index in range(len(proposer_models))]
    head_branch = branch_names[0]
    branch_workdirs = {
        name: f"{fanout_workdir_prefix}_{index}"
        for index, name in enumerate(branch_names)
    }

    def agent_builder(role: str, model: BaseChatModel, max_turns: int):
        return create_agent(
            role=role,
            model=model,
            tools=list(tools),
            prompt_context=prompt_context,
            output_schema=Proposal,
            response_format_strategy=response_format_strategy,
            max_turns=max_turns,
            announce_turns=announce_turns,
            reset_turns_when_reentrant=False,
            model_retries=model_retries,
            checkpointer=False,  # Reducer AgentState is managed by this graph
        )

    proposers = {
        name: agent_builder(proposer_role, model, max_proposer_turns)
        for name, model in zip(branch_names, proposer_models, strict=True)
    }
    reducer = agent_builder(proposer_role, proposer_models[0], max_reducer_turns)

    def invocation_instruction(state: FanoutReduceState) -> HumanMessage:
        """Return this invocation's required, not-yet-consumed instruction."""
        instruction = state.get("invocation_instruction")
        if instruction is None:
            raise ValueError(
                "invocation_instruction is required for every graph invocation"
            )
        return instruction

    def initialize(state: FanoutReduceState):
        """Prepare workdirs and open an initial round with no proposer history."""
        del state
        mkdir_result = run_shell.invoke(
            {"command": shlex.join(["mkdir", "-p", "--", *branch_workdirs.values()])}
        )
        if not isinstance(mkdir_result, Mapping) or mkdir_result.get("returncode") != 0:
            raise RuntimeError(f"Failed to create fan-out workdirs: {mkdir_result!r}")
        return {
            "branch_proposals": dict.fromkeys(branch_names),
            "proposal": None,
            "reducer_state": None,
        }

    def make_propose(branch: str):
        def propose(state: FanoutReduceState, config: RunnableConfig) -> dict:
            """Answer on this branch alone, unaware of the other branches."""
            result = proposers[branch].invoke(
                {
                    "messages": [
                        invocation_instruction(state).model_copy(update={"id": None}),
                        _workspace_message(branch_workdirs[branch]),
                    ],
                    "current_turn": 0,
                },
                config=_child_graph_config(config),
            )

            update: dict[str, Any] = {
                "branch_proposals": {branch: result.get("structured_response")},
            }
            if branch == head_branch:
                update["reducer_state"] = result
            return update

        return propose

    def route_entry(state: FanoutReduceState) -> Literal["initialize", "reduce"]:
        """Fan out only when there is no complete reducer answer to revise."""
        if state.get("proposal") is not None and state.get("reducer_state") is not None:
            return "reduce"
        return "initialize"

    def invoke_reducer(
        state: FanoutReduceState,
        config: RunnableConfig,
        *messages: HumanMessage,
    ) -> AgentState:
        """Continue the head model with a fresh per-invocation turn budget."""
        reducer_state_past = state.get("reducer_state") or {}
        reducer_state_current = {
            **reducer_state_past,
            "messages": list(reducer_state_past.get("messages", [])) + list(messages),
            "current_turn": 0,
        }
        return cast(
            AgentState,
            reducer.invoke(
                reducer_state_current,
                config=_child_graph_config(config),
            ),
        )

    def reduce(state: FanoutReduceState, config: RunnableConfig) -> dict:
        """Merge the initial fan-out, or revise the previous merged answer."""
        previous_proposal = state.get("proposal")
        if previous_proposal is not None:
            result = invoke_reducer(
                state,
                config,
                _current_proposal_message(previous_proposal),
                invocation_instruction(state),
            )
            revised = result.get("structured_response")
            return {
                "proposal": revised if revised is not None else previous_proposal,
                "reducer_state": result,
            }

        proposals = state.get("branch_proposals") or {}
        head_proposal = proposals.get(head_branch)
        other_proposals = [
            proposal
            for name in branch_names[1:]
            if (proposal := proposals.get(name)) is not None
        ]
        fallback = (
            head_proposal
            if head_proposal is not None
            else next(iter(other_proposals), None)
        )
        if fallback is None:
            return {}
        if head_proposal is not None and not other_proposals:
            return {"proposal": head_proposal}

        result = invoke_reducer(
            state,
            config,
            _other_proposals_message(other_proposals),
        )
        merged = result.get("structured_response")

        return {
            "proposal": merged if merged is not None else fallback,
            "reducer_state": result,
        }

    def finish(state: FanoutReduceState) -> dict:
        """Mark this invocation's instruction as consumed."""
        del state
        return {"invocation_instruction": None}

    stage = StateGraph(state_schema=FanoutReduceState)  # type: ignore[type-var]

    stage.add_node("initialize", initialize)
    for name in branch_names:
        stage.add_node(name, make_propose(name))
    stage.add_node("reduce", reduce)
    stage.add_node("finish", finish)

    stage.add_conditional_edges(START, route_entry)
    for name in branch_names:
        stage.add_edge("initialize", name)
        stage.add_edge(name, "reduce")
    stage.add_edge("reduce", "finish")
    stage.add_edge("finish", END)

    return stage.compile(checkpointer=checkpointer)
