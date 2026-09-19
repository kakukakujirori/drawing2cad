Guidelines:
- Read the current `interpretation`: each hypothesis's adopted candidate, the one with the highest confidence, states each finished shape, material/void meaning, position, orientation, size and termination; `datum` fixes the common model frame. Preserve these decisions while choosing CAD operations and their order.
- Each entry is one operation that leaves a solid. Put its modelling method in `verb`; a sketch or profile belongs in the operation that extrudes, revolves or sweeps it.
- Cite a feature parameter as `sem_main_bore.radius` or `sem_main_bore.center`, and a printed figure as `dim_bore_diameter.nominal_value` or `dim_hole_count.quantity`. The pipeline annotates these references with their current scalar, array or null values. Use the exact parameter names in the adopted candidate's `parameters`; null means unknown, never zero. Do not copy a known measurement into prose.
- Open the feature's evidence regions when a claim needs checking.
- Keep the complete plan to at most 25 entries. Name each operation `op_...` for what it does, retaining that identity across revisions.
- List operations in build order; each one basically changes the previous operation's result. When one does not, its `detail` says which earlier results it takes (e.g., for cut or union), or that it takes none and starts a new body.
- An operation's `semantics` names every `sem_` feature it helps build. A feature may take several operations and an operation may serve several features. Every adopted candidate needs an operation. Build and cite adopted candidates only; a rejected candidate is context, not part of the part.
- Account for the base volume, additions and cuts, then fillets and chamfers.
- Review the interpretation's concerns in `stage_reports.interpretation.concerns` and its null parameters. If construction requires a choice the interpretation has not established, make a provisional choice and report the affected `sem_` parameter and choice. Do not silently replace a stated value. Only the audit can open a ticket.
- Use `run_shell` and `load_image` to inspect source views when needed. Address applicable audit feedback and stay within the announced turn budget.

Example `detail`: "Cut a hole of radius sem_main_bore.radius at sem_main_bore.center, along sem_main_bore.axis, through the host plate."
