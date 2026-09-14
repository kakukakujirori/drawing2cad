## Reconstruction record
The pipeline repeats `interpretation -> operations -> coding + verification -> audit`.
The pipeline owns `$reconstruction_path`; never edit or print the whole file. `.input_drawings` is the immutable input-file manifest, not an interpreted artifact.
`.snapshots[-1]` is the current round: it holds `round`, `last_completed_stage`, `open_tickets`, `interpretation`, `operations`, `program_source`, and `verification`.
Inside a snapshot, `.interpretation.views[]` holds registered input files and cropped views with `view_` names; each view's `.dimensions[]` holds `dim_` readings. `.interpretation.features[]` holds `sem_` features with numeric `parameters`, `dimension_refs` and localized `evidence`. Operations are in `.operations.proposal[]`; code is the `.program_source` string.
An artifact that is `null` has not been committed in this round. In a revision, read that stage's baseline from `.snapshots[-2]`; null does not mean the previous artifact was empty. Upstream artifacts already committed in the current round take precedence.
Reasoning stages read their assigned tickets and earlier `responses`; audit reads every current ticket and its responses. Use filtered `jq -c` queries to read these and the artifacts needed for the stage.
Fetch individual `view_...`, `dim_...`, `sem_...`, `op_...`, or `ret_...` members by name; do not dump the whole history.
Parallelize only independent tool calls. After creating or changing a file, wait for that tool's result before reading or measuring the file.
