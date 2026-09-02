# Zero-shot Drawing2CAD

This codebase runs 2D drawings to 3D CAD conversion **zero-shot** using general-purpose VLMs such as GPT or Gemini.

### How to run

Generate:

```bash
# gpt5.6-sol
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
