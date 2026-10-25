You write a procedure for reading one property of a feature from an engineering drawing made of orthographic views.

The property: $question, $meaning. Its answers: $answers.

A reader follows your procedure for one feature at a time. The reader sees the drawing's views with the feature's region boxed and answers only questions about what is drawn: which lines there are and where, solid or dashed, where they end, how many circles, straight or curved. A program follows the reader's answers through your procedure to a conclusion. The reader is never asked the property itself, because readers who are asked it answer from habit instead of from the lines.

Write each question so that:
- it is answered by looking, not by inferring the 3D shape;
- it names views by their relation to the feature: the view along its axis (where a hole or cut is seen end-on) or a view across it;
- it has two to four short answers, one of which may be "cannot tell".

A node with id "hidden" asks "Are hidden lines drawn?" with answers "yes" and "no"; the program answers it from the drawing's settings, not the reader. Use it if the procedure depends on whether hidden lines are drawn.

Each answer leads to another node's id or to a conclusion written "=answer": one of $answers, or "=undecided" when the lines do not settle it. Prefer undecided to a guess: a wrong conclusion raises a false alarm against a correct reading.

Mistakes readers made on this property, with the lines that decided it:
$cases

Answer with one JSON object and nothing else, with at most 8 nodes:
{"applies": "what a feature must have for this procedure to apply", "root": "n1", "nodes": [{"id": "n1", "ask": "...", "answers": {"yes": "n2", "no": "=$first", "cannot tell": "=undecided"}}]}
