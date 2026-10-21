# Native coder + auditor baseline

`zeroshot.pipeline_native` with an auditor added, and nothing else. It isolates
what the staged pipeline's tools and middleware add over its coder-auditor topology.

- **Coder**: the native agent, with the same prompt, tools (`run_shell`, `load_image`) and inputs.
  It owes no interpretation file. The pipeline neither builds nor renders `model.py`.
- **Auditor**: the same two tools, the input drawing, and the coder's final
  message (`prompts/auditor.md`, `prompts/audit_request.md`). It may build and render
  the program itself. It answers in free text and ends with `VERDICT: ACCEPT` or `VERDICT: REJECT`;
  if there is no verdict, the audit counts as a rejection. Edits it makes to `model.py` are undone (`model_restored`).
- **Rounds**: after a rejection, the coder continues its own conversation with
  the report (`prompts/revision.md`). As in `workflow/coder_auditor.yaml`, at most
  `max_audits: 1` audits, `max_turns: 60` per coder round and `auditor_max_turns: 20`.
  A coder out of turns ends the run, as in the native baseline. The auditor gets
  the staged auditor's `TurnBudgetMiddleware`: each turn is announced and the last is answer-only,
  so every audit ends in a report.

When a response hits the output limit, both agents are retried with a reminder, as the native
generator is: the coder gets native's, and the auditor is asked for its report and verdict.

Settings live in `zeroshot/configs/native2.yaml`, which extends `native.yaml`.
Logging, `on_existing`, the retrospective (coder only), and scoring follow the
native README. `run.json` says `"pipeline": "native2"`. `events.jsonl` holds
both agents, split by `prompt` records naming the `agent`. It also holds `audit_verdict` and
`run_completed.coder_rounds`. The auditor's transcript is `audit_messages.json`.

```bash
python -m zeroshot.pipeline_native2 --multirun \
  model=gpt6_luna_codex \
  artifact_root=outputs_baselines2/luna6.0/test_eccv \
  on_existing=retry \
  sample.sample_id=$(ls data/test_eccv/target_step_ori | sed "s/\.step$//; s/.*/'&'/" | paste -sd,)
python -m zeroshot.evaluation.aggregate_run --run-dir outputs_baselines2/luna6.0/test_eccv
```

`aggregate_run` counts both agents' calls and tokens, and lists their tool calls
under `generator`.
