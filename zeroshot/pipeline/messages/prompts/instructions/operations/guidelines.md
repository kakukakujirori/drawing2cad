Guidelines:
- Every entry is one modelling operation that turns the part so far into the part after it. Say which one in `verb`. A profile is not an operation -- it belongs to the entry that extrudes, revolves or sweeps it.
- The hypothesis's `geometry` gives each feature's shape and size but never where it sits. Placing the features on the part is this stage's work, done by matching the evidence across views.
- Its `evidence` holds numbers read straight off the drawing, in the sheet coordinates of the view each reading names: $view_frame.
- Place an operation against the features around it, not against a coordinate system of your own: on the face of a feature, on the axis of one, through the width of one. The plan is about the solid, and where the sheet sits relative to the model is the coder's to settle once.
- **Do not copy a number out of the hypothesis.** Cite a 3D claim as `sem_<feature>.geo_<claim>.<parameter>` and a drawing reading as `sem_<feature>.ev_<reading>.<parameter>` -- for example `sem_main_bore.geo_cylinder.radius` or `sem_base_profile.ev_front_left_edge.start` -- and it is filled in for you before the coder reads it. The `sem_`, `geo_`, and `ev_` names and the parameter name are printed in the hypothesis. Write out only the numbers the hypothesis does not hold: a depth you chose, an offset you worked out, a clearance.
- Keep the plan, which is what your edits merge into and not what you write out, to at most 25 entries.
- Name each operation in `name`, beginning `op_` and carrying on in lower_snake_case, for what the step does: `op_base_plate`, `op_bore_through`, `op_fillet_top_edges`. The `op_` marks it as a step the way `sem_` marks a hypothesis feature, so a step named after the feature it builds still reads as the step. The name is how every later stage cites it, and it carries no position -- the order is worked out separately.
- Say what each operation waits on, in `depends_on`, and which hypothesis features it helps build, in `semantics`. Both hold stable names: a step waiting on `op_base_plate` and building `sem_main_bore` has `depends_on: ["op_base_plate"]` and `semantics: ["sem_main_bore"]`; an empty list says there are none. A feature may take several operations, and an operation may serve several. An operation may only wait on others in the same plan, and the waiting must not come round in a circle.
- The build order is worked out from `depends_on`, not from the order you list operations in, so list them however you reason and put each dependency where it belongs.
- Account for the whole part: the base volume, then what is added to and cut from it, then the fillets and chamfers. Every feature the hypothesis establishes needs an operation that builds it. A repeated feature is written as the operations it repeats, one entry each.
- Revise an existing entry to answer feedback about the operation it performs, keeping its name so that references to it stay true and so that it replaces that entry rather than adding another. An entry you add takes a new name of its own; no existing identity changes or shifts. An entry you leave out of `edits` stays exactly as it was.
- Where the hypothesis records an `open_question`, or leaves it open whether a region is added or removed material, choose one, record the choice, and continue.
- If the hypothesis looks wrong, or is missing a number you need, plan anyway: choose, then name the `sem_` member you doubt and your choice in your ticket response. Only the audit can open a ticket, so that response is the one place a doubt reaches it.
- Use `run_shell` and `load_image` to inspect the drawing or the perspective renders when the hypothesis leaves a dimension or a placement in doubt.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

A `detail` reads like these. The name, the verb, the dependencies and the features the step builds go in their own fields, so none of them is written into the sentence:
- "Extrude the front-view outline 25 mm along +z to form the base plate."
- "Cut a hole of sem_main_bore.geo_cylinder.radius through the plate at sem_main_bore.ev_front_circle.center, entering the +y face."

Plain numbers in them are allowed if they are not present in the semantic hypothesis. Anything it holds should be cited instead, like `sem_main_bore.geo_cylinder.radius` or `sem_main_bore.ev_front_circle.center`.

Cite a parameter under the name the feature states it by -- the ones you can see in its `geometry` and `evidence` above.

Stop calling tools when you answer, and write nothing around the answer itself.
