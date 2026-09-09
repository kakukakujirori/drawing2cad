## Reconstruction record
The pipeline repeats `drawings -> semantics -> operations -> coding + verification -> audit`.
The pipeline owns `$reconstruction_path`; never edit or print the whole file. `.input_drawings` is immutable.
`.snapshots[-1]` is the current round: it holds `round`, `last_completed_stage`, `open_tickets`, `drawings`, `semantics`, `operations`, `program_source`, and `verification`.
Inside a snapshot: `.drawings.sheets[]` holds sheets with `name,role,file,evidence,dimensions`; each sheet's `.evidence[]` holds `ev_` entries and `.dimensions[]` holds `dim_` entries. Features are in `.semantics.proposal[]`, operations in `.operations.proposal[]`, and code is the `.program_source` string.
An artifact that is `null` has not been committed in this round. In a revision, read that stage's baseline from `.snapshots[-2]`; null does not mean the previous proposal was empty. Upstream artifacts already committed in the current round take precedence.
Your instruction names your assigned tickets. Read those tickets, their earlier `responses`, and only the artifact this stage needs with filtered `jq -c` queries.
Fetch individual `sheet_...`, `sem_...`, `op_...`, or `ret_...` members by name; do not dump the whole history.
Finish tool work first, then return one structured submission by itself with exactly one response per assigned ticket and none for other tickets.
