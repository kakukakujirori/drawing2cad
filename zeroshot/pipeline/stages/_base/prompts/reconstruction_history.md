## Reconstruction record
The pipeline repeats `interpretation -> operations -> coding + verification -> audit`.
The pipeline owns `$reconstruction_path`; never edit or print the whole file. `.input_drawings` is the immutable input-file manifest, not an interpreted artifact.
`.snapshots[-1]` is the current round: it holds `round`, `last_completed_stage`, `open_tickets`, `interpretation`, `operations`, `program_source`, `verification`, and `stage_reports` keyed by completed stage.
Inside a snapshot, `.interpretation.views[]` holds registered input files and cropped views with `view_` names; each view's `.dimensions[]` holds `dim_` readings. `.interpretation.features[]` holds `sem_` features with numeric `parameters`, `dimension_refs` and localized `evidence`. Operations are in `.operations.proposal[]`; code is the `.program_source` string.
An artifact that is `null` has not been committed in this round. In a revision, read that stage's baseline from `.snapshots[-2]`; null does not mean the previous artifact was empty. Upstream artifacts already committed in the current round take precedence.

### Tickets
Round 0 has one ticket whose `subject.instruction` asks for the whole reconstruction. A later ticket's `subject` is an audit finding whose `revision_request` gives the `action`, the `targets` (members, or a whole stage when `name` is null), the `instruction`, and `proposed_names` for new members. `assigned_stages` runs from the targets' stage through coding; `responses` holds the answers of assigned stages that already finished.
Reasoning stages read their assigned tickets; audit reads every current ticket. Use filtered `jq -c` queries to read these and the artifacts needed for the stage.
A revision changes only what its tickets cover: the targets and the proposed names. A member may also change when it cites one of them, a member changed this round, or an earlier-stage member that cites one of them. Members cite through region views, `evidence`, `dimension_refs`, `semantics` and `detail` references; `ret_x` cites `op_x`. A whole-stage `modify` covers that stage. The pipeline compares each new artifact with `.snapshots[-2]` and refuses other changes the stage report does not explain.

### Stage reports
A reasoning stage answers with one summary per assigned ticket and a `stage_report`. A summary states the ticket's outcome, including upstream blockers or provisional interpretations needed to explain it.
- `remark`: additional concerns outside those answers, empty if none. A shared explanation may appear here once.
- `dimension_checks`: coding's per-dimension report; null in other stages.
- `unticketed_changes`: each member changed although no ticket covers it, with the reason. A rename lists both names. `{}` when tickets cover every change, and in round 0.
Read relevant `stage_reports` too; missing reports in old snapshots mean no report was recorded.
Fetch individual `view_...`, `dim_...`, `sem_...`, `op_...`, or `ret_...` members by name; do not dump the whole history.
Parallelize only independent tool calls. After creating or changing a file, wait for that tool's result before reading or measuring the file.
