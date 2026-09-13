Guidelines:
- Read the current `interpretation`: `features` states each finished shape, material/void meaning, position, orientation, size and termination; `datum` fixes the common model frame. Preserve these decisions while choosing CAD operations and their dependencies.
- Each entry is one operation that leaves a solid. Put its modelling method in `verb`; a sketch or profile belongs in the operation that extrudes, revolves or sweeps it.
- Cite a feature parameter as `sem_main_bore.radius` or `sem_main_bore.center`, and a printed figure as `dim_bore_diameter.nominal_value` or `dim_hole_count.quantity`. The pipeline annotates these references with their current scalar, array or null values. Use the exact parameter names in `features[].parameters`; null means unknown, never zero. Do not copy a known measurement into prose.
- View regions locate drawing evidence in the referenced view's own file. Their pixel/UV coordinates are not model XYZ. Use the feature's model parameters and datum for placement; open the evidence regions when a claim needs checking.
- Keep the complete plan to at most 25 entries. Name each operation `op_...` for what it does, retaining that identity across revisions.
- `depends_on` names the `op_` results this operation consumes. `semantics` names every `sem_` feature it helps build. A feature may take several operations and an operation may serve several features. Every interpreted feature needs an operation, and every cited feature must exist in the interpretation.
- The build order comes from `depends_on`, not the JSON list order. Dependencies must stay within the plan and must not form a cycle.
- Account for the base volume, additions and cuts, then fillets and chamfers. Use the interpretation's feature placement rather than selecting a new datum or independently relocating features.
- Write the whole plan to the JSON file. Preserve every entry your tickets do not affect, under the name it already holds.
- Review `interpretation.questions` and null parameters. If construction requires a choice the interpretation has not established, make a provisional choice and record the affected `sem_` parameter and choice in your ticket response. Do not silently replace a stated value. Only the audit can open a ticket.
- Use `run_shell` and `load_image` to inspect source views when needed. Address applicable audit feedback and stay within the announced turn budget.

Example `detail`: "Cut a hole of radius sem_main_bore.radius at sem_main_bore.center, along sem_main_bore.axis, through the host plate."
