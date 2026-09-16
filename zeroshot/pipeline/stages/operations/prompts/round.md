## Operation Round
Revise the operation plan for reconstruction round $current_round.

Tickets assigned to operations this round: $assigned_tickets

Maintain the complete plan in `$operations_output_path`. The first round seeds no file: create it, establishing every operation the current drawing interpretation takes. A later round seeds it with the plan the preceding round settled: change the operations your assigned tickets and the interpretation-stage responses recorded on them affect, and leave every other operation and its `op_...` name exactly as it stands, or give the reason for the change in `stage_report.unticketed_changes`. Its contents when you answer become this round's plan. A ticket that is not assigned to you was traced to a defect in the program itself; leave it to coding.

After a turn that changes the file, the pipeline validates it against the current interpretation and reports the result. Correct what it rejects and write again.

Once the file validates, finish tool work and return one `TicketAnswers`: exactly one operations-stage response for each assigned ticket and none for any other, each naming the concrete `op_...` entries you established, changed, or examined. Include any additional concerns in remark; do not repeat the plan.

OperationPlan JSON schema:
```json
$operations_schema
```
