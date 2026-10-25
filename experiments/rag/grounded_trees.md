You write a procedure that checks one claim of a reading of an engineering drawing against the drawing itself.

A reading describes a part's features with coordinates. A program draws one claim onto the drawing's orthographic views as a mark, and shows a reader each view with the mark, beside an enlarged cut-out around it. The reader answers only questions about what is drawn in relation to the mark: which lines lie inside it or along it, where they end, whether they are solid or dashed, whether they slant. A program follows the answers through your procedure to a conclusion and compares it with the claim. The reader never sees the claim in words and is never asked the property itself, because readers asked the property answer from habit instead of from the lines.

The mark: $mark

The procedure concludes one of: $conclusions, or "undecided" when the lines do not settle it. Prefer undecided to a guess: a wrong conclusion raises a false alarm against a correct reading.

Drawing conventions: visible edges are solid lines, hidden edges are dashed, and dimension, extension and centre lines are thin. A node with id "hidden" asks "Are hidden lines drawn?" with the answers "yes" and "no"; the program answers it from the drawing's settings. Use it where the procedure depends on whether hidden lines are drawn.

Lines that mislead readers about this kind of claim, which your questions must tell apart:
$confusers

Write each question so that:
- it asks one thing about the lines in relation to the mark, in at most 25 words;
- it has the answers "yes", "no" and "cannot tell", unless a short list of shapes is clearer;
- it names the mark and its parts as the mark's description does.

Each answer leads to another node's id or to a conclusion written "=conclusion".

Mistakes readers made on claims of this kind, with the lines that decided them:
$cases

Answer with one JSON object and nothing else, with at most $max_nodes nodes:
{"applies": "the claims this procedure checks", "root": "n1", "nodes": [{"id": "n1", "ask": "...", "answers": {"yes": "n2", "no": "=undecided", "cannot tell": "=undecided"}}]}
