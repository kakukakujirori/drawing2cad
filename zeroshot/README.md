# Zero-shot Drawing2CAD

This codebase runs 2D drawings to 3D CAD conversion **zero-shot** using general-purpose VLMs such as GPT or Gemini.

### How to run

Generate:

```bash
python -m zeroshot.run_pipeline --multirun \
    model=gpt5_6_luna_codex \
    artifact_root=outputs/baseline_luna_xhigh \
    on_existing=retry \
    sample.sample_id=$(ls data/test_vlm/target_step | sed 's/\.step//' | paste -sd,)
```

Evaluate:

```bash
python -m zeroshot.evaluation.aggregate_run \
    --run-dir outputs/baseline_luna_xhigh
```
