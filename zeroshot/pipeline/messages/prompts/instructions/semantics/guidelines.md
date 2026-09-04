Guidelines:

Reading the input
- The input message names each sheet you were given and says what it is. Analyse them with `run_shell` and `load_image` before you answer.
- Where a sheet arrives undivided, separate the views yourself by where the linework sits: they are not separated by layer or by file.
- Whether an edge is visible or hidden is how depth is read — whether a hole is through or blind, where a pocket stops. In a vector sheet the linetype says so (`Continuous` seen, `HIDDEN` behind material); in a raster sheet the dashes do.
- Numbers come from two places and they are not equally good. **A figure printed on the drawing is authoritative**: use it as given. A vector sheet also states every curve outright. Either way that is `given`.
- **Everything else you have to work out**, and that is `derived` however you did it -- a radius from a diameter, a step from two lengths, or a length taken off the linework. To take one off the linework, work out how many millimetres one unit of the sheet is, using a printed dimension whose linework you can identify, and measure the rest through that.
- A number nothing in the drawing supports is `assumed`, and saying so is worth more than a number that looks certain.
- Measure with code, never by eye. `run_shell` gives you the same tools for a raster sheet that `ezdxf` gives you for a vector one: read the image with OpenCV or numpy, find the linework, and compute the numbers. A coordinate guessed off a picture is worse than one you admit you do not have.
- Report every number in the drawing's own units. Pixels belong nowhere in your answer.
- Work out which 2D primitives in different views are one 3D entity seen more than once.

Interpreting the geometry
- Work outward from the base body, through the major features, to the local details such as rounds and chamfers, until nothing drawn is left unaccounted for.
- A curve in a view is evidence, not a conclusion — a spline silhouette is far more often the projection of a fillet than a freeform surface. Record what you saw in `evidence`, and what you claim in `geometry`.
- If you cannot pin a curve's parameters down, mark the entry `assumed` and say so in `open_question`. Saying you could not is worth more than a number nothing supports.

Filling in the answer
- The axes are not yours to choose. $view_frame. Report every number in the ones the drawing itself uses.
- One entry in `edits` per feature you are giving. Its `name` is its stable reference identity: begin it with `sem_` and continue in lower_snake_case, such as `sem_base_body` or `sem_main_bore`. Do not add a separate display label; make the reference name readable and explain the feature in `description`. Keep the name when revising the feature, since that is what says you are revising it rather than adding another. The base body is a feature like the others and comes first in the round that establishes it. Whether a feature is built by adding or removing material is the modelling plan's decision, not yours.
- A feature you give carries the `geometry` members you changed and the full list of evidence it cites: the geometry you leave out keeps what it had, and a claim you no longer want goes in `deleted` as `sem_<feature>.geo_<claim>`. `description` and `open_question` carry no name of their own, so state them whenever you give a feature.
- Give every member of a feature's `geometry` a stable `name`, unique within that list, beginning `geo_`, and every piece of evidence a stable `name`, unique within the whole hypothesis, beginning `ev_`. Later stages cite `sem_main_bore.geo_cylinder.radius` and `ev_front_circle.center`. A name is an identity, not the member's position in the list: keep it when revising the member and give a new member a new name.
- A feature is one thing that can be named and measured on its own: a plate is one prismatic body rather than four lines, and two rounds of different radius are two features rather than one rounded edge.
- `geometry` says what must be present in the finished solid: the `kind` that names the real face rather than one that merely resembles it — a rounded edge is a torus or a cylinder, a flat chamfer a plane — which `axis` it turns about, and how big it is.
- **`geometry` carries no position.** How big and which way it faces, not where it sits; the modelling plan places it from your evidence. A size you got wrong shows up in the next render, but a position you got wrong is inherited in silence.
- `evidence` is the drawing transcribed, and it stands outside the features: the view, the entity type as drawn, whether the linework was visible or hidden, where the numbers came from, and the entity's own numbers in millimetres on the sheet. A feature cites the evidence that supports it by name, so one entry can support several and none of them owns it. Every feature must cite at least one.
- Give an entry every number its entity takes, or give it none. An entry cited for what it proves rather than what it measures — a pair of hidden lines showing a bore runs through — takes no parameters, and that is complete. A half-transcribed entity is not.
- `description` explains the feature. No dimension belongs there that belongs in `geometry`.
- When you merge several models' answers into one, return a single `edits` list whose `sem_`, `geo_`, and `ev_` names remain unique in their respective scopes. Preserve an existing name where the same feature or member survives; rename only to resolve a real collision.

Working
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Analyse the drawing with `run_shell` and `load_image` before you answer. Submit your answer only once the analysis is complete, and write nothing around the answer itself.
