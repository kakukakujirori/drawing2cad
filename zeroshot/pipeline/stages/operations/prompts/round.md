Revise the operation plan for reconstruction round $current_round.

Tickets assigned to operations this round: $assigned_tickets

Maintain the complete plan in `$operations_output_path`. The first round seeds that file with `null`, so establish every operation the current drawing interpretation takes. A later round seeds it with the plan the preceding round settled: change the operations your assigned tickets and the interpretation-stage responses recorded on them affect, and leave every other operation and its `op_...` name exactly as it stands. The file is the whole plan and never a diff, so an operation you drop from it is deleted. A ticket that is not assigned to you was traced to a defect in the program itself; leave it to coding.

After a turn that changes the file, the pipeline validates it against the current interpretation and reports the result. Correct what it rejects and write again.

Return one `TicketAnswers` once the file validates: exactly one operations-stage response for each assigned ticket and none for any other, each naming the concrete `op_...` entries you established, changed, or examined. Do not repeat the plan in that answer.

OperationPlan JSON schema:
```json
$operations_schema
```
