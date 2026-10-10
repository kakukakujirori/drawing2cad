# Zero-shot Drawing2CAD

This codebase runs 2D drawings to 3D CAD conversion **zero-shot** using general-purpose VLMs such as GPT or Gemini.

### How to run

Generate:

```bash
conda activate drawing2cad

# gpt6-luna
python -m zeroshot.run_pipeline --multirun \
    model=gpt6_luna_codex \
    artifact_root=outputs/gpt6_luna \
    on_existing=retry \
    workflow=continued \
    sample.sample_id=$(ls data/test_vlm/target_step_ori | sed 's/\.step//' | paste -sd,)

# glm5.3-flash
python -m zeroshot.run_pipeline --multirun \
    model=glm5.3_flash_openrouter \
    artifact_root=outputs/glm5.3_flash \
    on_existing=retry \
    workflow=continued \
    sample.sample_id=$(ls data/test_vlm/target_step_ori | sed 's/\.step//' | paste -sd,)
```

Rescore finished samples without rerunning the agents. A rerun with
`on_existing=retry` skips completed samples, so it does not rescore them. Each
sample is scored against `<target-dir>/<sample_id>.step`. Directories without
`events.jsonl` or a target are skipped. For f360, pass
`data/ortho2cad/test100_gt_steps_f360`:

```bash
python -m zeroshot.evaluation.run_scoring \
    --run-dir outputs/gpt6_luna/*/ \
    --target-dir data/test_vlm/target_step_ori
```

Evaluate:

```bash
python -m zeroshot.evaluation.aggregate_run \
    --run-dir outputs/gpt6_luna
```

After each sample, `StepScorer` writes mesh IoU, squared Chamfer/Hausdorff,
reference ECCV F1 and official Ortho2CAD IoU to `score.json`. The default GT is
`data/test_vlm/target_step_ori`, whose dimensions match the DXF drawings. Main
metrics share the GT-derived scale (`reference_extent=1.8`) and maximum-IoU
alignment over 24 cube rotations; Ortho2CAD uses its own official preprocessing.
Rotations within 1e-3 IoU of the best are the same geometry, so ECCV F1 takes the
best of them: a symmetric part's seams otherwise make its F1 depend on the pose.
Rotations within 1e-3 IoU of the best are ties. ECCV takes the best of them,
because a symmetric part's B-Rep seams make its F1 depend on the tied pose.

`aggregate_run` reports AUC-TR and valid-only mean/median CD, in IterCAD's unit
(GT bbox diagonal = 1) rather than `reference_extent`. Generation failures
stay in the AUC denominator; a missing CD from an evaluator error leaves these
aggregates undefined. Distance metrics never fill missing values with zero.
Sampling, mesh tolerances and AUC thresholds are configurable under
`evaluation.scorer` and recorded with each score. Existing logs require rescoring
before they can be combined with the new metric protocol.

## Single-agent baseline

Coder + Auditor construction:

```bash
# gpt6-luna
python -m zeroshot.run_pipeline --multirun \
    model=gpt6_luna_codex \
    artifact_root=outputs/gpt6_luna_single \
    on_existing=retry \
    workflow=single \
    sample.sample_id=$(ls data/test_vlm/target_step_ori | sed 's/\.step//' | paste -sd,)

# glm5.3-flash
python -m zeroshot.run_pipeline --multirun \
    model=glm5.3_flash_openrouter \
    artifact_root=outputs/glm5.3_flash_single \
    on_existing=retry \
    workflow=single \
    sample.sample_id=$(ls data/test_vlm/target_step_ori | sed 's/\.step//' | paste -sd,)
```

## Interactive Debug

```bash
python -m interactive_debug \
  --run outputs/glm5.3_flash/xxx/checkpoints.sqlite \
  --stage {interpretation, coding, audit} \
  --round 000
```
