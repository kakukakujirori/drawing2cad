You read an engineering drawing in $projection projection.$hidden The views follow, each with numbered boxes. Each item below names a kind of feature and the boxes where it is drawn; an item's number marks its boxes in every view.

For each item, answer each of its questions from the drawing alone, by the lines that decide it:
$questions
Edge asks how the feature meets the face it opens on, not the corners of its outline.

If an item covers several features of its kind and they answer differently, answer "mixed".

Items:
$items

Answer with one JSON object and nothing else, one entry per item and question:
{"answers": [{"item": 1, "question": "through or blind", "answer": "blind", "lines": "the lines that decide it"}]}
