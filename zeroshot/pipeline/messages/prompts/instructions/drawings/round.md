Read the input drawing for reconstruction round $current_round.

Tickets assigned to the drawing stage this round: $assigned_tickets

The pipeline has seeded the complete working `DrawingSource` at $drawing_output_path. In the initial round it contains only what the run was handed; separate every page into views and transcribe them. In a later assigned round it contains the accepted reading from the preceding round; change only what your assigned tickets require. A ticket that is not assigned to you was traced to a defect downstream of the drawing, so change nothing on its account.

Write the entire updated `DrawingSource` back to $drawing_output_path. Then return one `DrawingSubmission` containing exactly one drawing-stage response for each assigned ticket and none for any other. Each response must name the concrete `sheet_...` entries you established, changed, or examined. The JSON file is the drawing deliverable; do not put sheets in the structured submission.

The file must validate against this `DrawingSource` schema:

```json
$drawing_schema
```


$guidelines
