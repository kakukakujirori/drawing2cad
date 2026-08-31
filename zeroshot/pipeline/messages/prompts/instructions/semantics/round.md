Produce the complete semantic hypothesis for reconstruction round $current_round.

Tickets assigned to semantics this round: $assigned_tickets

For the initial round, analyse the input drawing and establish every geometric feature and relationship that the later modelling plan must reproduce. In later rounds, revise that complete hypothesis wherever your assigned tickets require it while preserving stable names for unchanged features. A ticket that is not assigned to you was traced to a defect downstream of semantics; leave it to the stage that owns it and do not change the hypothesis on its account.

Return one complete `SemanticSubmission`: put the complete `SemanticHypothesis` in `deliverable`, and include exactly one semantics-stage response for each assigned ticket and none for any other. Each response must name the concrete `sem_...` entries you established, changed, or examined.


$guidelines
