## Reconstruction record
The pipeline repeats `interpretation -> operations -> coding + verification -> audit`.
The pipeline owns `$reconstruction_path`; never edit or print the whole file. `.input_drawings` is the immutable input-file manifest, not an interpreted artifact.
`.snapshots[-1]` is the current round: it holds `round`, `last_completed_stage`, `open_tickets`, `interpretation`, `operations`, `program_source`, and `verification`.
Inside a snapshot, `.interpretation.views[]` holds registered input files and cropped views with `view_` names; each view's `.dimensions[]` holds `dim_` readings. `.interpretation.features[]` holds `sem_` features with numeric `parameters`, `dimension_refs` and localized `evidence`. Operations are in `.operations.proposal[]`; code is the `.program_source` string.
An artifact that is `null` has not been committed in this round. In a revision, read that stage's baseline from `.snapshots[-2]`; null does not mean the previous artifact was empty. Upstream artifacts already committed in the current round take precedence.
Your instruction names your assigned tickets. Read those tickets, their earlier `responses`, and the artifacts needed for this stage with filtered `jq -c` queries.
Fetch individual `view_...`, `dim_...`, `sem_...`, `op_...`, or `ret_...` members by name; do not dump the whole history.
Finish tool work first, then return one structured submission by itself with exactly one response per assigned ticket and none for other tickets.
