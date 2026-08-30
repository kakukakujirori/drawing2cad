Implement the complete CadQuery program for reconstruction round $current_round.

The pipeline-owned reconstruction history is at `$reconstruction_path`. Inspect its latest snapshot, which contains this round's completed semantic hypothesis, operation plan, open tickets, and prior stage responses. Earlier snapshots remain available there when you need to understand what changed, persisted, or regressed. Do not edit that file yourself.

Create or inspect the program at `$output_path`. Implement every current operation and verify the resulting solid before you stop. In a revision round, preserve code that remains correct and update everything required by the current snapshot and its tickets.

Return one complete `CodingSubmission`. Its `deliverable` must be null because the pipeline captures `model.py` through verification. Include exactly one coding-stage response for every open ticket, naming the concrete `ret_...` results or `result` that you implemented, changed, or examined.


$guidelines
