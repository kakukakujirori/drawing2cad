Guidelines:
- Begin with the base volume: the workplane it is built on, its profile, and the direction and distance it is extruded or revolved.
- Then the major additive and subtractive features, each with its profile, its depth, and its position on the part. State whether a hole is through or blind.
- Then local details: fillets, chamfers, patterns. Name the edges or faces they apply to in terms of features already planned.
- Order the entries so each one refers only to features an earlier entry introduced: a fillet after the edge it rounds, a pattern after its seed.
- Keep the plan to at most 25 entries. Each entry covers one feature from the hypothesis and states the high-level operations that materialise it. Revise an existing entry to answer feedback about the feature it produces.
- Where the hypothesis leaves it open whether a region is added or removed material, choose one, record the choice, and continue.
- Use `run_shell` and `load_image` to inspect the drawing or the perspective renders when the hypothesis leaves a dimension or a placement in doubt.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Write one operation per entry, in build order, each naming the feature it produces and the dimensions it uses, for example:
- "Extrude the 80 x 40 front-view outline 25 mm along +z to form the base plate."
- "Cut a 10 mm diameter through hole at (20, 15) on the top face."

Stop calling tools when you answer, and write nothing around the answer itself.
