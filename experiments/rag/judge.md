You check how a model read an engineering drawing, against the real part.

You get, in this order:
1. The drawing the model read.
2. The real part (gt.png) and what the model built from its reading (build.png). Both are drawn in the model's frame, with its axes shown, from four directions; each panel's title says from where.
3. Their difference (diff.png): the real part in translucent grey, the material the build lacks in red (M1, M2, ...) and the material it adds in blue (E1, E2, ...).

Each piece of the difference, with its volume as a share of the real part's:
$components

The model's reading: its views, with the model axes pointing right and up in each, and its features, with descriptions and parameters (lengths in mm):
$reading

Do two things.

A. For each piece, name the features of the reading it comes from, and its cause:
- reading: the reading gets that feature wrong (its description or its parameters), or lacks a feature the part has
- building: the reading is right there, but the build does not follow it
- negligible: a sliver or a small rounding that no reading needs to name
- unclear: you cannot tell

B. List each error of the reading. Give:
- feature: its name in the reading, or null if the reading lacks the feature altogether
- decision: the question the reading answers wrong, from the list below, with the reading's answer (read) and the part's (truth). Questions without answers take null for both. Pick the first question in the list that fits; a feature wrong on two questions has two errors.
$questions
- reading: what the reading says
- truth: what the part has
- cue: the lines of the drawing that decide it, by the direction of the view to the feature (along its axis, across it) and whether they are solid or hidden
- components: the pieces it explains

Judge the 3D shape, not sizes, unless a size changes the shape, such as a blind depth read as through or a wall too thick to leave a cavity. Every piece whose cause is reading belongs to at least one error. Write reading, truth and cue in general words, with no numbers, axis letters, view names or feature names, so that a reader of another drawing can use them.

Answer with one JSON object and nothing else:
{"components": [{"id": "M1", "features": ["..."], "cause": "reading"}], "errors": [{"feature": "...", "decision": {"question": "through or blind", "read": "through", "truth": "blind"}, "reading": "...", "truth": "...", "cue": "...", "components": ["M1"]}]}
