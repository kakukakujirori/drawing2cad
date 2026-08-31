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
├─ `schema_version`
├─ `run_id`
└─ `snapshots`: ReconstructionSnapshot[]
   ├─ `open_tickets`: Ticket[]
   │  ├─ `ticket_id`
   │  ├─ `subject`: BootstrapWork | AuditFinding
   │  │  ├─ BootstrapWork: `instruction`
   │  │  └─ AuditFinding: `name`, `observation`, `evidence`, `backtraces[]`
   │  │     └─ backtrace: `hops[]` plus `revision_request`
   │  │        ├─ hop: `effect`, `cause`, `rationale`
   │  │        └─ request: `action`, `targets`, `instruction`, `proposed_names`
   │  ├─ `assigned_stages`: ReasoningStage[]
   │  └─ `responses`: TicketResponse[]
   │     ├─ `ticket_id`
   │     ├─ `stage`
   │     └─ `summary`
   ├─ `round`
   ├─ `last_completed_stage`
   ├─ `semantics`: SemanticHypothesis | null
   │  ├─ `proposal[]`: `name`, `description`, `geometry[]`, `evidence[]`, `open_question`
   │  │  ├─ geometry: `name`, `kind`, `source`, `axis`, `parameters[]`
   │  │  └─ evidence: `name`, `view`, `entity`, `edge_style`, `parameters[]`
   │  └─ `rationale`
   ├─ `operations`: OperationPlan | null
   │  ├─ `proposal[]`: `name`, `verb`, `detail`, `depends_on`, `semantics`
   │  └─ `rationale`
   ├─ `program_source`: str | null
   └─ `verification`: VerifyOutputResult | null
      └─ `verification_id`, `status`, `returncode`, `stdout`, `stderr`, `executor_error`, `shape`
```

Inspect only what the current task needs. Do not print the whole history file, and never `cat` it: one round's hypothesis alone runs to tens of thousands of tokens, and the file holds every round. Read an index of stable names first, then fetch by name the one member you need. `jq -c` keeps a record to a line.

```bash
# Where the round stands, and who owns each ticket.
jq -c '.snapshots[-1] | {round, last_completed_stage, tickets: [.open_tickets[] | {ticket_id, assigned_stages}]}' '$reconstruction_path'

# What one ticket asks, by the id your instruction gave you.
jq -c '.snapshots[-1].open_tickets[] | select(.ticket_id == "ticket_001_wrong_bore") | .subject.backtraces[].revision_request' '$reconstruction_path'

# An index of names. Read this before any artifact body.
jq -c '[.snapshots[-1].semantics.proposal[] | {name, geo: [.geometry[].name], ev: [.evidence[].name]}]' '$reconstruction_path'
jq -c '[.snapshots[-1].operations.proposal[] | {name, verb, depends_on, semantics}]' '$reconstruction_path'

# What each stage reported, including any doubt it raised about the stage above it.
jq -c '[.snapshots[-1].open_tickets[].responses[] | {ticket_id, stage, summary}]' '$reconstruction_path'

# Then one member in full, by a name the index gave you.
jq -c '.snapshots[-1].semantics.proposal[] | select(.name == "sem_main_bore")' '$reconstruction_path'
jq -c '.snapshots[-1].operations.proposal[] | select(.semantics | index("sem_main_bore"))' '$reconstruction_path'

# The program by the operation it implements, not the whole file.
jq -r '.snapshots[-1].program_source' '$reconstruction_path' | grep -n 'ret_main_bore'
```

Use `.snapshots[-2]` only when at least two snapshots exist, and compare only the upstream artifact your stage reads. A later stage's artifact is still null in the current round, and iterating over it fails.

- Semantics: your tickets, and the preceding semantics when revising.
- Operations: the preceding semantics against the current one.
- Coding: the preceding operations against the current ones.
- Audit: compare rounds only to decide whether a defect persisted or regressed.

Diff the index rather than the bodies, sorted so that the diff is about content and not about order:

```bash
diff -u <(jq -S '[.snapshots[-2].operations.proposal[] | {name, verb, detail}] | sort_by(.name)' '$reconstruction_path') <(jq -S '[.snapshots[-1].operations.proposal[] | {name, verb, detail}] | sort_by(.name)' '$reconstruction_path')
```

A bootstrap subject describes the initial build; an audit-finding subject describes an observed defect and traces it to the requested revision. The pipeline sets `assigned_stages` from that revision root and runs it through coding, because a corrected artifact has to be carried down into the program. Your instruction names the tickets assigned to you: answer each of those, and leave every other ticket to the stage that owns it. Earlier responses on a ticket explain what upstream stages already changed. Do not edit the history file yourself.
