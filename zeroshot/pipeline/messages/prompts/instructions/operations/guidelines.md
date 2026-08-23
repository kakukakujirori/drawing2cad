Guidelines:
- The hypothesis's `geometry` gives each feature's shape and size but never where it sits. Placing the features on the part is this stage's work, done by matching the evidence across views.
- Its `evidence` holds numbers read straight off the drawing, in the sheet coordinates of the view each reading names: $view_frame. Convert them into model coordinates yourself.
- Begin with the base volume: the workplane it is built on, its profile, and the direction and distance it is extruded or revolved.
- Then the major additive and subtractive features, each with its profile, its depth, and its position on the part. State whether a hole is through or blind.
- Then local details: fillets, chamfers, patterns. Name the edges or faces they apply to in terms of features already planned.
- Keep the plan to at most 25 entries. One entry is one operation.
- Say what each operation waits on, in `depends_on`, and which hypothesis features it helps build, in `semantics`. Both are lists of plain numbers rather than names -- an operation waiting on op3 and building sem7 has `depends_on: [3]` and `semantics: [7]` -- and an empty list says there are none.
- A feature may take several operations, and an operation may serve several. An operation may only wait on others in the same plan, and the waiting must not come round in a circle.
- The build order is worked out from `depends_on`, not from the order you list operations in, so list them however you reason and put each dependency where it belongs.
- Revise an existing entry to answer feedback about the operation it performs, keeping its id so that references to it stay true.
- Where the hypothesis records an `open_question`, or leaves it open whether a region is added or removed material, choose one, record the choice, and continue.
- Use `run_shell` and `load_image` to inspect the drawing or the perspective renders when the hypothesis leaves a dimension or a placement in doubt.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

An operation reads like these, with the dependencies and the features it
builds carried in their own fields rather than written into the sentence:
- "Extrude the 80 x 40 front-view outline 25 mm along +z to form the base plate."
- "Cut a 10 mm diameter hole through the plate at x=20, z=15, entering the +y face."

Stop calling tools when you answer, and write nothing around the answer itself.
