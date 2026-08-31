Produce the complete operation plan for reconstruction round $current_round.

Plan every operation needed to construct the current semantic hypothesis. In a revision round, reconsider every operation affected by the tickets and by the recorded semantics-stage responses; preserve stable names for unchanged operations.

Return one complete `OperationSubmission`: put the complete `OperationPlan` in `deliverable`, and include exactly one operations-stage response for every open ticket. Each response must name the concrete `op_...` entries you established, changed, or examined.


$guidelines
