You check a reading of an engineering drawing against the real part, feature by feature.

You get the drawing's views, each feature's evidence boxed and numbered, then the real part (gt.png) drawn from four directions.$hidden

The reading's features:
$features

Below are properties, each with a procedure of questions about what is drawn. For each item and each property:
- applies: whether the real part, within the item's boxed region, has what the property asks about. Judge the real part, not the reading.
- If it applies, truth: the real part's answer to the property.
- If it applies, observations: an answer to every question of the property's procedure about the drawing as drawn, even questions a path would skip. Use exactly one of the answers offered.

Properties and procedures:
$procedures

Answer with one JSON object and nothing else, one entry per item and property:
{"answers": [{"item": 1, "property": "through or blind", "applies": "yes", "truth": "blind", "observations": {"n1": "yes"}}, {"item": 1, "property": "edge", "applies": "no"}]}
