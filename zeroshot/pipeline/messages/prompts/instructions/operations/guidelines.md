Guidelines:
- Every entry is one modelling operation that turns the part so far into the part after it. Say which one in `verb`. A profile is not an operation -- it belongs to the entry that extrudes, revolves or sweeps it.
- The hypothesis's `geometry` gives each feature's shape and size but never where it sits. Placing the features on the part is this stage's work, done by matching the evidence across views.
- Its `evidence` holds numbers read straight off the drawing, in the sheet coordinates of the view each reading names: $view_frame.
- Place an operation against the features around it, not against a coordinate system of your own: on the face of a feature, on the axis of one, through the width of one. The plan is about the solid, and where the sheet sits relative to the model is the coder's to settle once.
- **Do not copy a number out of the hypothesis.** Cite it as `sem<id>.<parameter>` -- `sem7.radius`, or `sem1.torus.major_radius` where one feature claims more than one geometry -- and it is filled in for you before the coder reads it. Write out only the numbers the hypothesis does not hold: a depth you chose, an offset you worked out, a clearance.
- Keep the plan to at most 25 entries.
- Say what each operation waits on, in `depends_on`, and which hypothesis features it helps build, in `semantics`. Both are lists of plain numbers rather than names -- an operation waiting on op3 and building sem7 has `depends_on: [3]` and `semantics: [7]` -- and an empty list says there are none. A feature may take several operations, and an operation may serve several. An operation may only wait on others in the same plan, and the waiting must not come round in a circle.
- The build order is worked out from `depends_on`, not from the order you list operations in, so list them however you reason and put each dependency where it belongs.
- Account for the whole part: the base volume, then what is added to and cut from it, then the fillets and chamfers. Every feature the hypothesis establishes needs an operation that builds it. A repeated feature is written as the operations it repeats, one entry each.
- Revise an existing entry to answer feedback about the operation it performs, keeping its id so that references to it stay true.
- Where the hypothesis records an `open_question`, or leaves it open whether a region is added or removed material, choose one, record the choice, and continue.
- Use `run_shell` and `load_image` to inspect the drawing or the perspective renders when the hypothesis leaves a dimension or a placement in doubt.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

A `detail` reads like these, with the verb, the dependencies and the features
it builds carried in their own fields rather than written into the sentence.
The first works its extrusion distance out from the views and so writes it; the
second takes a radius the hypothesis states and so cites it:
- "Extrude the front-view outline 25 mm along +z to form the base plate."
- "Cut a hole of sem4.radius through the plate at x=20, z=15, entering the +y face."

Cite a parameter under the name the feature states it by -- the ones you can
see in its `geometry` and `evidence` above.

Stop calling tools when you answer, and write nothing around the answer itself.
