Below is a reading of an engineering drawing: its datum and its features, each with a description and parameters (lengths in model units, positions in the model frame the datum defines).

Turn it into marks: the geometry of each feature, as the reading has it, in the model frame.
- hole: a hole, slot, pocket, recess or window cut into material. One mark per instance: a feature with four holes gives four marks. Give its centre on its axis (any point on the axis, at the hole's centre across), its axis as a unit vector, its size across the axis as two directions perpendicular to the axis with the width along each (a round hole has its diameter along both), its entry and far end as positions along the axis (the axis coordinate where it opens and where it ends), and whether the reading has it through or blind.
- body: a solid or shell body, block, plate, boss, rib or flange: the box it fills, as the low and high corner, and whether the reading calls it solid (true) or hollow (false).
- cavity: an internal hollow or the inside of a shell: the box it fills, as the low and high corner.

Use the parameters; where the reading gives a position only in words, derive it from the datum and the other parameters. Leave out a feature you cannot place. Do not judge the reading: write what it says.

The reading:
$reading

Answer with one JSON object and nothing else:
{"holes": [{"feature": "sem_...", "center": [0, 0, 0], "axis": [0, 1, 0], "across": [[[1, 0, 0], 4.0], [[0, 0, 1], 4.0]], "entry": 0.0, "end": 10.0, "claim": "through"}],
 "bodies": [{"feature": "sem_...", "low": [0, 0, 0], "high": [10, 10, 10], "solid": true}],
 "cavities": [{"feature": "sem_...", "low": [1, 1, 1], "high": [9, 9, 9]}]}
