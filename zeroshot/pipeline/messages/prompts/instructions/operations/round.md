Revise the operation plan for reconstruction round $current_round.

Tickets assigned to operations this round: $assigned_tickets

In the initial round the plan is empty, so plan every operation the current semantic hypothesis takes. In a later round the plan the preceding round settled stands as it is, and you change every operation your assigned tickets and the semantics-stage responses recorded on them affect: give those operations, and leave every other one out so that it keeps what it had. A ticket that is not assigned to you was traced to a defect in the program itself; leave it to coding.

Return one `OperationSubmission`: the operations you changed in `edits`, whatever you dropped in `deleted`, and exactly one operations-stage response for each assigned ticket and none for any other. Each response must name the concrete `op_...` entries you established, changed, or examined.


$guidelines
