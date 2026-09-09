Rules:
- Transcribe 2D marks as drawn and leave their 3D meaning to semantics.
- Split an undivided page by linework position, retain the original `full_page`, save each crop separately, and leave the separated page itself without evidence.
- Measure rasters with code, then fit mm/unit using `calculate_drawing_scale` with printed dimensions and matching pixel lengths.
- Each view has its own millimetre UV frame. Put its raster origin at the lower-left corner of the bottom-left pixel, not at that pixel's centre: pixel `(r,c)` has corner `(c, h - r - 1)` and centre `(c + 0.5, h - r - 0.5)` before scaling. $view_frame.
- Preserve visible, hidden, centerline, and phantom styles; DXF layer `0` does not separate views, while linetype distinguishes styles.
- Arc/ellipse endpoints lie on the curve and sweep counterclockwise; point lists are flattened x,y pairs.
- Record each printed figure once and exactly. It supplies size, not position, so never move measured geometry merely to fit a nominal.
- Use unique stable lower_snake_case `sheet_`, `ev_`, and `dim_` names. Keep perspective sheets qualitative, without invented scale or primitives.
