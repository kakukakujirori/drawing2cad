Interpret the drawing for round $current_round. Assigned tickets: $assigned_tickets.

Write the complete artifact to `$interpretation_output_path`. That file is always seeded in full: on the first round with the input files already registered as views and `datum` left as `"???"`, and on revisions with the interpretation accepted last round. Settle the datum, read the views, and add what the drawing states. Retain unaffected entries and stable names while resolving assigned tickets; omitted entries are removed.

Successful verification fills the derived fields in that same file. Continue editing it and use the automatic feedback to correct errors and inconsistent measurements.

After the first pass across the views, save a provisional artifact by turn 5, with unresolved issues in questions. Use subsequent image inspections to resolve those issues and complete the major geometry, then resubmit. Reserve turns to read the automatic validation and calibration feedback and correct errors. A valid file confirms the contract and scale checks; it does not confirm the 3D interpretation.

After the current artifact validates, finish tool work and return one TicketAnswers with exactly one response per assigned ticket and none for other tickets. Describe only changes present in that artifact.

DrawingInterpretation JSON schema:
```json
$interpretation_schema
```
