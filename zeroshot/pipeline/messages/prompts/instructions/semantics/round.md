Revise the semantic hypothesis for reconstruction round $current_round.

Tickets assigned to semantics this round: $assigned_tickets

In the initial round the hypothesis is empty, so analyse the input drawing and establish every geometric feature and relationship the later modelling plan must reproduce. In a later round the hypothesis the preceding round settled stands as it is, and you change what your assigned tickets require: give those features, and leave every other feature out so that it keeps what it had. A ticket that is not assigned to you was traced to a defect downstream of semantics; leave it to the stage that owns it and change nothing on its account.

Return one `SemanticSubmission`: the features you changed in `edits`, whatever you dropped in `deleted`, and exactly one semantics-stage response for each assigned ticket and none for any other. Each response must name the concrete `sem_...` entries you established, changed, or examined.


$guidelines
