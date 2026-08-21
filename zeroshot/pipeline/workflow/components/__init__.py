"""The graph shapes a stage can be built from, one shape per module.

`agent` is the ReAct loop every stage ultimately runs. The other two compose it
into the multi-agent shapes a stage may need: `fanout_reduce` answers once with
every model and merges through the head, and `proposer_reviewer` bounces one
answer between a proposer and a reviewer until it settles.
"""

from .agent import AgentState, create_agent
from .compact import compact_transcript
from .fanout_reduce import FanoutReduceState, create_fanout_reduce_graph
from .fanout_reduce import Proposal as FanoutReduceProposal
from .proposer_reviewer import (
    Proposal,
    ProposerReviewerState,
    Review,
    create_proposer_reviewer_loop,
)

__all__ = [
    "AgentState",
    "FanoutReduceProposal",
    "FanoutReduceState",
    "Proposal",
    "ProposerReviewerState",
    "Review",
    "compact_transcript",
    "create_agent",
    "create_fanout_reduce_graph",
    "create_proposer_reviewer_loop",
]
