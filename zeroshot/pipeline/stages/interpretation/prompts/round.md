Interpret the drawing for round $current_round. Assigned tickets: $assigned_tickets.

Write the complete artifact to `$interpretation_output_path`. On the first round no file is seeded: create it from the supplied images. On revisions the previous accepted interpretation is supplied in full at that path. Retain unaffected entries and stable names while resolving assigned tickets; omitted entries are removed.

Successful verification fills the derived fields in that same file. Continue editing it and use the automatic feedback to correct errors and inconsistent measurements.

After the first pass across the views, save a provisional artifact by turn 5, with unresolved issues in questions. Use subsequent image inspections to resolve those issues and complete the major geometry, then resubmit. Reserve turns to read the automatic validation and calibration feedback, correct errors, and submit one TicketAnswers after the current artifact validates. A valid file confirms the contract and scale checks; it does not confirm the 3D interpretation.

DrawingInterpretation JSON schema:
```json
$interpretation_schema
```
