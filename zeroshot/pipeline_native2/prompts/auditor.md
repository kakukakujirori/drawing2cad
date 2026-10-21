The CadQuery Python script /work/model.py reconstructs the part shown in the provided 2D engineering drawing. It assigns the final solid, with lengths in mm, to a variable named result. Length dimensions on the drawing are in millimetres, whatever their decimals or the part's apparent size; use them as printed, never converted or rescaled. Check the solid it builds against the drawing. You may run the script and inspect the solid in any way, such as by rendering it from the drawing's view directions. Do not modify /work/model.py; save the files you create under /work/audit/.

Report every material mismatch between the solid and the drawing: what the drawing shows, what the solid has, and where. Quantify the difference when the drawing supports a measurement. State what is wrong, not how to fix it; the author of the script decides the correction. A successful build alone does not establish geometric correctness.

End your answer with one line: `VERDICT: ACCEPT` when the solid matches the drawing in all material respects, otherwise `VERDICT: REJECT`. Stay within the announced turn budget.

### Which view is which?
$projection_placement In an L of three unturned views, the corner view is front. Explicit view labels take precedence. Do not move front to another view because its silhouette is more informative or more familiar, and do not call a conventional arrangement free-style for that reason. Treat the arrangement as free-style only when labels and the agreement between views do not fix the roles; then choose roles whose projections agree with each other.
