Guidelines:

Reading the input
- Analyse the input DXF drawing using python libraries such as `ezdxf`. You can render DXF views to PNG using `ezdxf draw <DXF_PATH> -o <OUTPUT_PNG>` via `run_shell`, and inspect them with `load_image`.
- The views are not separated by layer: effectively all geometry sits on layer `0`. Separate Front, Top and Right by clustering the entities by position on the sheet.
- Linetype is what tells you whether an edge is visible. `Continuous` linework is an edge seen from that direction, `HIDDEN` linework an edge behind material — which is how depth is read, such as whether a hole is through or blind.
- Read the curve definitions, not the picture. Every curve entity carries the numbers that define it; extract them and carry them into your answer unchanged.
- Work out which 2D primitives in Front, Top and Right are one 3D entity seen three ways.

Interpreting the geometry
- Work outward from the base body, through the major features, to the local details such as rounds and chamfers, until nothing drawn is left unaccounted for.
- A curve in a view is evidence, not a conclusion — a spline silhouette is far more often the projection of a fillet than a freeform surface. Record what you saw in `evidence`, and what you claim in `geometry`.
- If you cannot pin a curve's parameters down, mark the entry `assumed` and say so in `open_question`.

Filling in the answer
- The axes are not yours to choose. $view_frame. Report every number in the ones the drawing itself uses.
- One entry in `edits` per feature you are giving. Its `name` is its stable reference identity: begin it with `sem_` and continue in lower_snake_case, such as `sem_base_body` or `sem_main_bore`. Do not add a separate display label; make the reference name readable and explain the feature in `description`. Keep the name when revising the feature, since that is what says you are revising it rather than adding another. The base body is a feature like the others and comes first in the round that establishes it. Whether a feature is built by adding or removing material is the modelling plan's decision, not yours.
- A feature you give carries the `geometry` and `evidence` members you changed, and no others: the ones you leave out keep what they had, and a member you no longer want goes in `deleted` as `sem_<feature>.geo_<claim>` or `sem_<feature>.ev_<reading>`. `description` and `open_question` carry no name of their own, so state them whenever you give a feature.
- Give every member of a feature's `geometry` a stable `name`, unique within that list, beginning `geo_`, and every member of its `evidence` a stable `name`, unique within that list, beginning `ev_`. For example, later stages cite `sem_main_bore.geo_cylinder` and `sem_main_bore.ev_front_circle`. A name is an identity, not the member's position in the list: keep it when revising the member and give a new member a new name.
- A feature is one thing that can be named and measured on its own: a plate is one prismatic body rather than four lines, and two rounds of different radius are two features rather than one rounded edge.
- `geometry` says what must be present in the finished solid: the `kind` that names the real face rather than one that merely resembles it — a rounded edge is a torus or a cylinder, a flat chamfer a plane — which `axis` it turns about, and how big it is.
- **`geometry` carries no position.** How big and which way it faces, not where it sits; the modelling plan places it from your evidence. A size you got wrong shows up in the next render, but a position you got wrong is inherited in silence.
- `evidence` is the drawing transcribed: the view, the entity type as drawn, whether the linework was visible or hidden, and the entity's own numbers in sheet coordinates. Exact geometry belongs here, because it is read rather than inferred. Every feature must cite at least one reading.
- `description` explains the feature. No dimension belongs there that belongs in `geometry`.
- When you merge several models' answers into one, return a single `edits` list whose `sem_`, `geo_`, and `ev_` names remain unique in their respective scopes. Preserve an existing name where the same feature or member survives; rename only to resolve a real collision.

Working
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Stop calling tools when you answer, and write nothing around the answer itself.
