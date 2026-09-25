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

## Single-agent baseline

Coder + Auditor construction:

```bash
# gpt5.6-luna
python -m zeroshot.run_pipeline --multirun \
    model=gpt5.6_luna_codex \
    artifact_root=outputs/gpt5.6_luna_single \
    on_existing=retry \
    workflow=single \
    sample.sample_id=$(ls data/test_vlm/target_step | sed 's/\.step//' | paste -sd,)

# glm5.3-flash
python -m zeroshot.run_pipeline --multirun \
    model=glm5.3_flash_openrouter \
    artifact_root=outputs/glm5.3_flash_single \
    on_existing=retry \
    workflow=single \
    sample.sample_id=$(ls data/test_vlm/target_step | sed 's/\.step//' | paste -sd,)
```

## Interactive Debug

```bash
python -m interactive_debug \
  --run outputs/glm5.3_flash/xxx/checkpoints.sqlite \
  --stage {interpretation, operations, coding, audit} \
  --round 000
```