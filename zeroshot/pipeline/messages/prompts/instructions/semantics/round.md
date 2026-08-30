Produce the complete semantic hypothesis for reconstruction round $current_round.

The pipeline-owned reconstruction history is at `$reconstruction_path`. Inspect its latest snapshot and address every open ticket. Earlier snapshots remain available there when you need to understand what changed, persisted, or regressed. Do not edit that file yourself.

For the initial round, analyse the input drawing and establish every geometric feature and relationship that the later modelling plan must reproduce. In later rounds, revise that complete hypothesis wherever the open tickets or earlier stage responses require it while preserving stable names for unchanged features.

Return one complete `SemanticSubmission`: put the complete `SemanticHypothesis` in `deliverable`, and include exactly one semantics-stage response for every open ticket. Each response must name the concrete `sem_...` entries you established, changed, or examined.


$guidelines
