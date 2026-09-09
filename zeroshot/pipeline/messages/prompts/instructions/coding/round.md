Implement the complete CadQuery program for reconstruction round $current_round.

Tickets assigned to coding this round: $assigned_tickets

Create or inspect the program at `$coding_output_path`. Implement every current operation and verify the resulting solid before you stop. In a revision round, preserve code that remains correct and update everything required by the current snapshot and your assigned tickets.

Write a working draft to that path before half the turn budget is spent, then use its automatic verification feedback to develop it. If a kernel operation keeps failing, inspect its geometric preconditions and try another construction of the intended feature. Do not silently skip the operation.

Return one `CodingSubmission`. It carries your ticket responses and nothing else: you revise the program in the workspace and the pipeline captures it through verification. Include exactly one coding-stage response for each assigned ticket and none for any other, naming the concrete `ret_...` results or `result` that you implemented, changed, or examined.


$guidelines
