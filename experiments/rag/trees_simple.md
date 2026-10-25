Below is a procedure for reading one property of a feature from an engineering drawing: a tree of questions about what is drawn, each answer leading to another question or to a conclusion.

Readers answer long questions badly. Rewrite the procedure to reach the same conclusions, so that every question:
- asks one thing about one view, in at most 20 words;
- has the answers "yes", "no" and "cannot tell", unless a count or a short list of shapes is clearer;
- names the view by its relation to the feature: the view along its axis, or a view across it.

Split a question that asks several things into several nodes. Keep the node "hidden" as it is if the procedure has it; the program answers it. Keep "=undecided" wherever the procedure has it, and add it wherever a short question leaves the case open.

The procedure, for $question ($answers):
$tree

Answer with one JSON object and nothing else, in the same form, with at most $max_nodes nodes:
{"applies": "...", "root": "n1", "nodes": [{"id": "n1", "ask": "...", "answers": {"yes": "n2", "no": "=undecided", "cannot tell": "=undecided"}}]}
