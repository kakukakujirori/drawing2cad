# Zero-shot Drawing2CAD

This codebase runs 2D drawings to 3D CAD conversion **zero-shot** using general-purpose VLMs such as GPT or Gemini.

### How to run

Generate:

```bash
conda activate drawing2cad

# gpt5.6-luna
python -m zeroshot.run_pipeline --multirun \
    model=gpt5.6_luna_codex \
    artifact_root=outputs/gpt5.6_luna \
    on_existing=retry \
    workflow=continued \
    sample.sample_id=$(ls data/test_vlm/target_step | sed 's/\.step//' | paste -sd,)

# glm5.3-flash
python -m zeroshot.run_pipeline --multirun \
    model=glm5.3_flash_openrouter \
    artifact_root=outputs/glm5.3_flash \
    on_existing=retry \
    workflow=continued \
    sample.sample_id=$(ls data/test_vlm/target_step | sed 's/\.step//' | paste -sd,)
```

Evaluate:

```bash
python -m zeroshot.evaluation.aggregate_run \
    --run-dir outputs/gpt5.6_luna
```

### Interpretation pipeline

The graph runs `interpretation -> operations -> coding -> audit`. The interpreter
writes the complete `interpretation.json` and returns `TicketAnswers`.
Operations similarly write the complete `operations.json` and return ticket
responses. Each verified file replaces that stage's complete artifact.
Ticket summaries describe their own outcomes; `remark` carries additional concerns.
Reports are saved under `.snapshots[-1].stage_reports` in `reconstruction.json`
for downstream stages and audit to read, including after resume. Audit reviews
each current defect ticket and relates unresolved defects to new findings;
the initial bootstrap work is read but is not carried into subsequent rounds.
Coding also reports `dimension_checks` for every registered dimension ID. Submission
validation enforces complete, nonblank coverage; audit checks the claims against
the final geometry. Unknown or unverified dimensions must be explained, not omitted.

After a JSON or referenced-image change, automatic verification checks the files,
calibrates each raster view from its dimensions, and reports errors, scale status,
inlier count/total, and outlier `dim_` names. Successful verification writes the
enriched artifact back to `interpretation.json`; failed validation preserves the
submitted file. The same enriched artifact is stored in `reconstruction.json`
at `.snapshots[-1].interpretation` and in the verification attempt directory.

For debugging, `workspace/attempts/round_NNN/interpretation/NNN/` also keeps the
pre-validation bytes as `_interpretation_raw.json` and full diagnostics as
`_interpretation_validation_log.json`. These files are not advertised in model
instructions or feedback. They remain discoverable through directory listings;
the underscore is a naming convention, not an access restriction.

Feature parameters can be referenced in operation details as
`sem_main_bore.radius` or `sem_main_bore.center`; printed dimensions use
`dim_bore_diameter.nominal_value`. Audit targets use the `interpretation` stage
with `view_`, `dim_`, or `sem_` names. An omitted feature can be requested directly
at that stage with an `add` finding and a null target name.

Input and rendered files use `DrawingView`. The first interpretation is seeded
with the registered input views and `datum: "???"`. Set the datum and add readings,
preserving each input's name, file, role and self-referencing full-file region;
pictorial inputs may be omitted. Create crops only for newly separated views.
Inputs require unique `view_` names and DXF, PNG or JPEG files. Old reconstruction
histories using `DrawingSource`, or separate `drawings` and `semantics` snapshots,
cannot be resumed directly.
Native DXF verification requires explicit `workflow.dxf_mm_per_unit` metadata
keyed by the registered `view_` names; it does not assume native units are mm.
For example, input `view_front` uses metadata key `view_front`. The interpreter
receives each original DXF's native origin, physical conversion and full-file
UV bounds. Additional DXF files need their own configured `view_` keys and use
their own geometry-bounding-box origins. Raster crops need no DXF factor.
