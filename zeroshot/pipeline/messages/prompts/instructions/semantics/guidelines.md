Guidelines:
- Analyse the input DXF drawing using python libraries such as `ezdxf`. You can render DXF views to PNG using `ezdxf draw <DXF_PATH> -o <OUTPUT_PNG>` via `run_shell`, and inspect them with `load_image`.
- Front, Top and Right are separately available by specifying the corresponding DXF layers.
- Perform cross-view correspondence matching: verify how 2D primitives in Front, Top, and Right views correspond to single 3D entities.
- Identify primary base geometry first, then major additive/subtractive features, then local details (fillets, chamfers, threads, hole specifications).
- Account for base volumes, additive features, subtractive features, holes, cutouts, bosses, flanges, fillets, chamfers, patterns, and symmetry.
- If feedback from a review or audit step is present in the transcript, address every point it raises.
- Your turn budget is announced in the transcript as `[turn n/N]`. Turns increment by using tools.

Write one self-contained entry per feature, for example:
- "Primary base body: <base volume and overall bounding dimensions>"
- "Feature 1: <additive or subtractive feature, location, dimensions>"

Stop calling tools when you answer, and write nothing around the answer itself.
