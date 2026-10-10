## Reconstruction workflow

You take part in reconstructing a 3D CAD model from a multi-view engineering drawing. The goal is agreement between the CAD geometry and the source drawing.

The pipeline refines its deliverables through repeated rounds of two stages:

- `coding`: read the source drawing into an interpretation (drawing views such as front, top and right, dimension readings, and the part's 3D features), then write an executable CadQuery program from the drawing and that interpretation. Automatic verification executes the program and projects the resulting solid onto the drawing's views.
- `audit`: compare the interpretation and the projections of the resulting solid with the source drawing, and report each mismatch with its cause: the interpretation or the program.

The auditor may open revision tickets for another round. The round instructions name your stage and its files. Edit only those files.

## Reconstruction record

`$reconstruction_path` is the pipeline-managed JSON record of inputs and round snapshots.
Never edit it or print the whole file. Use filtered `jq -c` queries to fetch the required fields and members by name.

- `.input_drawings`: immutable source-file manifest, before interpretation.
- `.snapshots[-1]`: current round.
- `.snapshots[-2]`: previous round; refer to it for revisions.

A member is a named item of the interpretation: `datum`, `view_...`, `dim_...` or `sem_...`. Each snapshot holds:

- `round`, `last_completed_stage`: progress.
- `open_tickets`: this round's work (see Tickets below).
- `interpretation.views[]`: registered inputs and crops (`view_`); each view's `dimensions[]` holds its dimension readings (`dim_`).
- `interpretation.features[]`: hypothetical 3D features (`sem_`) with numeric `parameters`, `dimension_refs` and localized `evidence`.
- `program_source`: CadQuery source code.
- `verification`: execution, rendering and drawing-comparison reports.
- `stage_reports`: reports keyed by completed stage.

Each round starts with null artifacts. A completed coding stage fills `.snapshots[-1]`; earlier snapshots stay unchanged. In rounds after 0, the previous round's results are in `.snapshots[-2]`.

### Tickets

Round 0 has one ticket for the whole reconstruction. For revisions, the pipeline turns each audit finding into a new ticket. Coding fixes what each ticket describes and answers every ticket. The pipeline validates and records the answers before the audit. The audit reviews every ticket; unresolved and newly found defects become the next round's tickets.

Ticket fields:

- `ticket_id`: stable `ticket_` identifier.
- `subject`: the initial reconstruction task (`instruction`) in round 0; an audit finding in later rounds.
- `subject.cause`: where the defect comes from: `interpretation` for the drawing reading, `coding` for the program.
- `subject.targets`: the interpretation members the defect concerns. For an omission, the view where the drawing shows it.
- `subject.revision_request`: what is wrong.
- `responses`: coding's answer, once coding has finished.
- `evidence_renders`: paths to images of `subject.evidence`, in order; regions are marked with red boxes or cropped out.

### Stage reports

Coding answers through `TicketAnswers`: `responses` maps ticket IDs to free-text outcomes, including provisional interpretations and what could not be resolved. `stage_report.concerns` holds further unresolved issues and provisional choices, one `concern_...` entry each, `{}` if none. The auditor reviews each.
