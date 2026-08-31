Produce the complete operation plan for reconstruction round $current_round.

Tickets assigned to operations this round: $assigned_tickets

Plan every operation needed to construct the current semantic hypothesis. In a revision round, reconsider every operation affected by your assigned tickets and by the semantics-stage responses recorded on them; preserve stable names for unchanged operations. A ticket that is not assigned to you was traced to a defect in the program itself; leave it to coding.

Return one complete `OperationSubmission`: put the complete `OperationPlan` in `deliverable`, and include exactly one operations-stage response for each assigned ticket and none for any other. Each response must name the concrete `op_...` entries you established, changed, or examined.


$guidelines
