## Coding Round
Reconstruct the part in the input drawing as a CadQuery program for round $current_round.

Create or inspect the program at `$coding_output_path`. Read the views and printed dimensions yourself, build the whole part, and verify the resulting solid before you stop. In a revision round, preserve code that remains correct and fix every audit finding below.

Write a working draft to that path before half the turn budget is spent, then use its automatic verification feedback to develop it. If a kernel operation keeps failing, inspect its geometric preconditions and try another construction of the intended feature. Do not silently skip the feature.

Finish tool work and return one `CodingReport`. Revise the program in the workspace; the pipeline captures it through verification.

Audit findings to address this round:
$audit_findings
