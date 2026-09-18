## Coding Round
Implement the complete CadQuery program for reconstruction round $current_round.

Tickets assigned to coding this round: $assigned_tickets

Create or inspect the program at `$coding_output_path`. Implement every current operation and verify the resulting solid before you stop. In a revision round, preserve code that remains correct and update everything required by the current snapshot and your assigned tickets. Give the reason for any other `ret_...` change in `stage_report.unticketed_changes`.

Write a working draft to that path before half the turn budget is spent, then use its automatic verification feedback to develop it. If a kernel operation keeps failing, inspect its geometric preconditions and try another construction of the intended feature. Do not silently skip the operation.

Finish tool work and return one `TicketAnswers` with your ticket responses, one `stage_report.concerns` entry per remaining concern those responses do not explain, and `dimension_checks`. Revise the program in the workspace; the pipeline captures it through verification. Include exactly one coding-stage response for each assigned ticket and none for any other, naming the concrete `ret_...` results or `result` that you implemented, changed, or examined.

Current dimension readings (all registered IDs, including those the plan does not cite):
```json
$dimension_inventory
```
