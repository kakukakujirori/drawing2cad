Guidelines:

Reading the input
- The drawing has already been read for you. `drawings` in the current reconstruction snapshot holds the input pages and the views read from them, with `ev_` entries for drawn entities and `dim_` entries for printed figures. Work from that: it is what your `evidence` cites, and the numbers in it are the ones the later stages build against.
- The original sheets are still yours to open. The input message names each of them, and `run_shell` and `load_image` reach them. Go back to a sheet when an entry looks wrong or when something you can see on the sheet has no entry at all, and say so in `open_question` rather than quietly writing a number of your own into `geometry`.
- `edge_style` is how depth is read. A `visible` entry is an edge seen from that direction, a `hidden` one an edge behind material — which is what says whether a hole is through or blind. A `centerline` or `phantom` entry marks an axis or a reference, not an edge of the part.
- Work out which entries on different sheets are one 3D entity seen from several directions. That correspondence is the work of this stage.

Interpreting the geometry
- Work outward from the base body, through the major features, to the local details such as rounds and chamfers, until nothing drawn is left unaccounted for.
- An entry is evidence, not a conclusion — a spline silhouette is far more often the projection of a fillet than a freeform surface. Cite what was read in `evidence`, and state what you claim in `geometry`.
- If the entries do not pin a curve down, say what is left open in `open_question` rather than choosing for them.

Filling in the answer
- One entry in `edits` per feature you are giving. Its `name` is its stable reference identity: begin it with `sem_` and continue in lower_snake_case, such as `sem_base_body` or `sem_main_bore`. Do not add a separate display label; make the reference name readable and explain the feature in `description`. Keep the name when revising the feature, since that is what says you are revising it rather than adding another. The base body is a feature like the others and comes first in the round that establishes it. Whether a feature is built by adding or removing material is the modelling plan's decision, not yours.
- A feature you give carries the `geometry` claims you changed, and no others: the ones you leave out keep what they had, and a claim you no longer want goes in `deleted` as `sem_<feature>.geo_<claim>`. `evidence` is a citation list rather than a list of members, so give the whole of it whenever you give the feature, and so state `description` and `open_question` too.
- Give every claim in a feature's `geometry` a stable `name`, unique within that list, beginning `geo_`, so that later stages can cite `sem_main_bore.geo_cylinder`. A name is an identity, not the claim's position in the list: keep it when revising the claim and give a new claim a new name. The `ev_` and `dim_` names are not yours to invent; they belong to the drawing.
- A feature is one thing that can be named and measured on its own: a plate is one prismatic body rather than four lines, and two rounds of different radius are two features rather than one rounded edge.
- `geometry` says what must be present in the finished solid: the `kind` that names the real face rather than one that merely resembles it — a rounded edge is a torus or a cylinder, a flat chamfer a plane — which `axis` it turns about, and how big it is.
- **`geometry` carries no position.** How big and which way it faces, not where it sits; the modelling plan places it from your evidence. A size you got wrong shows up in the next render, but a position you got wrong is inherited in silence.
- `evidence` names the entries and printed figures this feature rests on, by the `ev_` and `dim_` names the drawing gives them. Every feature must cite at least one, and two features may cite the same entry. A number you cannot trace to an entry or figure is a hole in the reading: report it rather than fill it.
- `description` explains the feature. No dimension belongs there that belongs in `geometry`.
- When you merge several models' answers into one, return a single `edits` list whose `sem_` and `geo_` names remain unique in their respective scopes. Preserve an existing name where the same feature or claim survives; rename only to resolve a real collision.

Working
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Read `drawings` before you answer, and open the sheets themselves where it is not enough. Submit your answer only once the analysis is complete, and write nothing around the answer itself.
