# """Own the two existing interpretation agents before combining their execution.

# The workflow adapts its state channels to these methods and commits their
# submissions. This module owns tools, schemas, drawing feedback and inference.
# """

# from collections.abc import Mapping, Sequence
# from dataclasses import dataclass
# from typing import Any

# from langchain_core.runnables import RunnableConfig
# from langchain_core.tools import BaseTool

# from zeroshot.pipeline.messages import DrawingSource
# from zeroshot.pipeline.messages.contracts.reconstruction import (
#     DrawingSubmission,
#     ReconstructionSnapshot,
#     SemanticSubmission,
#     tickets_assigned_to,
# )
# from zeroshot.pipeline.messages.contracts.stages import PipelineStage
# from zeroshot.pipeline.stages._base.prompt import RoundInstructions
# from zeroshot.pipeline.tools import create_calculate_drawing_scale_tool
# from zeroshot.pipeline.verification import AttemptStore, DrawingVerifier
# from zeroshot.pipeline.workflow.components.agent import (
#     AgentBuilder,
#     CompiledGraph,
#     invoke_agent,
# )
# from zeroshot.pipeline.workflow.middleware.verify_on_write import (
#     VerifyOnWriteMiddleware,
# )


# @dataclass(frozen=True)
# class SemanticStages:
#     drawings_agent: CompiledGraph
#     semantics_agent: CompiledGraph
#     drawing_verifier: DrawingVerifier
#     drawing_middleware: VerifyOnWriteMiddleware
#     instructions: RoundInstructions
#     input_after_compaction: bool

#     def run_drawings(
#         self,
#         *,
#         snapshot: ReconstructionSnapshot,
#         baseline: DrawingSource,
#         previous: Mapping[str, Any],
#         validation_error: str | None,
#         config: RunnableConfig,
#     ) -> dict[str, Any]:
#         if validation_error is None:
#             self.drawing_verifier.reset(baseline)
#             self.drawing_middleware.reset()
#         assigned = tickets_assigned_to(snapshot.open_tickets, PipelineStage.DRAWINGS)
#         if not assigned:
#             return {"stage_submission": DrawingSubmission.unchanged()}

#         instruction = self.instructions.build(
#             "drawings",
#             drawing=snapshot.drawings or baseline,
#             current_round=snapshot.round,
#             assigned_tickets=", ".join(ticket.ticket_id for ticket in assigned),
#             validation_error=validation_error,
#             include_input=(not previous or self.input_after_compaction),
#         )
#         result = invoke_agent(self.drawings_agent, previous, instruction, config)
#         return {
#             "drawings_state": result,
#             "stage_submission": result.get("structured_response"),
#         }

#     def run_semantics(
#         self,
#         *,
#         snapshot: ReconstructionSnapshot,
#         previous: Mapping[str, Any],
#         validation_error: str | None,
#         config: RunnableConfig,
#     ) -> dict[str, Any]:
#         if snapshot.last_completed_stage is not PipelineStage.DRAWINGS:
#             raise RuntimeError("semantics requires an integrated drawing")

#         assigned = tickets_assigned_to(snapshot.open_tickets, PipelineStage.SEMANTICS)
#         if not assigned:
#             return {"stage_submission": SemanticSubmission.unchanged()}

#         assert snapshot.drawings is not None  # Guaranteed by the snapshot contract.
#         instruction = self.instructions.build(
#             "semantics",
#             drawing=snapshot.drawings,
#             current_round=snapshot.round,
#             assigned_tickets=", ".join(ticket.ticket_id for ticket in assigned),
#             validation_error=validation_error,
#             include_input=(not previous or self.input_after_compaction),
#         )
#         result = invoke_agent(self.semantics_agent, previous, instruction, config)
#         return {
#             "semantics_state": result,
#             "stage_submission": result.get("structured_response"),
#         }


# def create_semantic_stages(
#     *,
#     drawings_agent_builder: AgentBuilder,
#     semantics_agent_builder: AgentBuilder,
#     tools: Sequence[BaseTool],
#     instructions: RoundInstructions,
#     attempt_store: AttemptStore,
#     drawing_filename: str,
#     input_after_compaction: bool,
# ) -> SemanticStages:
#     verifier = DrawingVerifier(
#         workdir=instructions.workdir,
#         attempt_store=attempt_store,
#         artifact_presenter=instructions.artifact_presenter,
#         source_filename=drawing_filename,
#     )
#     middleware = VerifyOnWriteMiddleware(
#         verifier,
#         refusal=(
#             "The drawing artifact is not ready to submit. Inspect the rendered "
#             "views, correct drawing.json if needed, and submit only after the "
#             "current version has been shown back to you and validates."
#         ),
#         require_feedback_before_submit=True,
#     )
#     return SemanticStages(
#         drawings_agent=drawings_agent_builder(
#             tools=[*tools, create_calculate_drawing_scale_tool()],
#             prompt_context=instructions.prompt_context,
#             output_schema=DrawingSubmission,
#             extra_middleware=[middleware],
#         ),
#         semantics_agent=semantics_agent_builder(
#             tools=tools,
#             prompt_context=instructions.prompt_context,
#             output_schema=SemanticSubmission,
#         ),
#         drawing_verifier=verifier,
#         drawing_middleware=middleware,
#         instructions=instructions,
#         input_after_compaction=input_after_compaction,
#     )
