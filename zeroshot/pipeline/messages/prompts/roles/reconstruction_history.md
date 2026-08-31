## Reconstruction history

This reconstruction runs as one repeated pipeline:

`semantics -> operations -> coding + verification -> audit`

- Semantics produces a complete `SemanticHypothesis`.
- Operations turns it into a complete dependency-aware `OperationPlan`.
- Coding writes `model.py`; verification executes it and records a `VerifyOutputResult`.
- Audit checks the completed, immutable round. An accepted report ends the run; findings become the open tickets of a new round.

The pipeline, not an agent, owns `$reconstruction_path`. It stores one `ReconstructionRun`. During a round the pipeline replaces the final snapshot after each completed reasoning stage. A new round starts with null deliverables, while every earlier snapshot remains available for comparison.

```text
ReconstructionRun
|- `schema_version`
|- `run_id`
`- `snapshots`: ReconstructionSnapshot[]
   |- `open_tickets`: Ticket[]
   |  |- `ticket_id`
   |  |- `subject`: BootstrapWork | AuditFinding
   |  |  |- BootstrapWork: `instruction`
   |  |  `- AuditFinding: `name`, `observation`, `evidence`, `backtraces[]`
   |  |     `- backtrace: `hops[]` plus `revision_request`
   |  |        |- hop: `effect`, `cause`, `rationale`
   |  |        `- request: `action`, `targets`, `instruction`, `proposed_names`
   |  `- `responses`: TicketResponse[]
   |     |- `ticket_id`
   |     |- `stage`
   |     `- `summary`
   |- `round`
   |- `last_completed_stage`
   |- `semantics`: SemanticHypothesis | null
   |  |- `proposal[]`: `name`, `description`, `geometry[]`, `evidence[]`, `open_question`
   |  |  |- geometry: `name`, `kind`, `source`, `axis`, `parameters[]`
   |  |  `- evidence: `name`, `view`, `entity`, `edge_style`, `parameters[]`
   |  `- `rationale`
   |- `operations`: OperationPlan | null
   |  |- `proposal[]`: `name`, `verb`, `detail`, `depends_on`, `semantics`
   |  `- `rationale`
   |- `program_source`: str | null
   `- `verification`: VerifyOutputResult | null
      `- `verification_id`, `status`, `source`, `returncode`, `stdout`, `stderr`, `executor_error`, `shape`
```

Inspect only what the current task needs. Do not print the whole history file. Useful starting points are:

```bash
jq '.snapshots[-1] | {round, last_completed_stage, open_tickets}' '$reconstruction_path'
jq '.snapshots[-1].semantics' '$reconstruction_path'
jq '.snapshots[-1].operations' '$reconstruction_path'
jq -r '.snapshots[-1].program_source' '$reconstruction_path'
```

Use `.snapshots[-2]` only when at least two snapshots exist. Compare only the upstream artifact relevant to your stage, using stable sorted JSON where useful:

- Semantics: inspect the current tickets and the preceding semantics when revising.
- Operations: compare preceding and current semantics.
- Coding: compare preceding and current operations.
- Audit: compare rounds only when deciding whether a defect persisted or regressed.

For example:

```bash
diff -u <(jq -S '.snapshots[-2].operations' '$reconstruction_path') <(jq -S '.snapshots[-1].operations' '$reconstruction_path')
```

Read the latest open tickets before acting. A bootstrap subject describes the initial build; an audit-finding subject describes an observed defect and traces it to the requested revision. Each reasoning stage must answer every ticket, and earlier responses on it explain what upstream stages changed. Do not edit the history file yourself.
