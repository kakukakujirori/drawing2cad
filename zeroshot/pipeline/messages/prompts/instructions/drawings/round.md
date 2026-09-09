Transcribe the drawing for round $current_round. Assigned tickets: $assigned_tickets.
`$drawing_output_path` is seeded from the input in round 0 and from the preceding accepted transcription in a revision; then change only what assigned tickets require.

Work in this order:
1. Inspect the seed and source; establish views and raster scale.
2. Before half the turn budget is spent, write a schema-valid working draft instead of waiting for every measurement.
3. Save new crop files beside the output JSON, outside the read-only input directory. Add evidence in batches; after every write, open and inspect the generated views against the source.
4. Reserve enough turns for a correction and final visual check. Submit only after the current file passes verification and its latest renders have been inspected.

Compact JSON contract; every shown key is required and extra keys are rejected:
- root `{"sheets":[sheet,...]}`; sheet `{name,role,label,crop_of,scale,file,evidence,dimensions}` with `name=sheet_*`, `role=front|back|top|bottom|left|right|section|detail|isometric|perspective|full_page|unknown`, and `crop_of=null|{sheet,box:[u0,v0,u1,v1]}`.
- evidence `{name,entity,edge_style,parameters,source}` with `name=ev_*`, `edge_style=visible|hidden|centerline|phantom|other`, same-sheet `dim_*` names in `source`. `parameters` is a list of `{name,values}` objects; `values` is always a numeric list (scalar `[x]`, point `[u,v]`).
- entity parameters: line=`start,end`; arc=`center,radius,start,end`; circle=`center,radius`; ellipse=`center,major_axis,minor_radius,start,end`; spline=`control_points,degree,knots`; polyline=`vertices`.
- dimension `{name,kind,text,nominal,quantity,note}` with `name=dim_*` and `kind=linear|diameter|radius|angular`.

One complete evidence entry:
```json
{"name":"ev_outline","entity":"line","edge_style":"visible","parameters":[{"name":"start","values":[0,0]},{"name":"end","values":[10,0]}],"source":[]}
```

`DrawingSubmission.responses` may summarize only sheets in the latest verified file; it is never a substitute transport for unwritten transcription. Return exactly one `DrawingSubmission`, alone, after all tool work.

$guidelines
