# Native coder + auditor baseline

`zeroshot.pipeline_native` with an auditor and the staged pipeline's turn budgets.
It isolates what the staged pipeline's reconstruction tools and validation add
over its coder-auditor topology.

- **Coder**: the native agent, with the same prompt, tools (`run_shell`, `load_image`) and inputs.
  It owes no interpretation file. The pipeline neither builds nor renders `model.py`.
- **Auditor**: the same two tools, the input drawing, and the coder's final
  message (`prompts/auditor.md`, `prompts/audit_request.md`). It may build and render
  the program itself. It answers in free text and ends with `VERDICT: ACCEPT` or `VERDICT: REJECT`;
  a missing verdict is retried. Edits it makes to
  `model.py` are undone (`model_restored`).
- **Rounds**: after a rejection, the coder continues its own conversation with
  the report (`prompts/revision.md`). As in `workflow/coder_auditor.yaml`, at most
  `max_audits: 1` audits, `max_turns: 60` per coder round and `auditor_max_turns: 20`.
  Both agents use a local copy of `TurnBudgetMiddleware`: each turn is announced,
  the penultimate turn warns to finish tool work, and the last is answer-only.
- **Grace turns**: an unfinished stage resumes its full state and conversation
  with `retry_turns: 10` turns left, up to `max_stage_validation_retries: 3` times,
  matching `workflow/coder_auditor.yaml`. Completion requires a final answer and
  a nonempty `model.py` for the coder, or a verdict for the auditor. An explicit
  `REJECT` opens a coder round rather than retrying the audit. A new round starts
  with fresh counters. The maximum is 90 coder turns per round and 50 per audit;
  early incomplete answers keep their remaining turns, rewinding only when fewer
  than 10 remain. If the middleware stops the agent for budget exhaustion after
  the retries, the run ends with `status: turn_limit_exceeded`. If the model
  answers but the required file or verdict is still missing, it ends with
  `status: stage_validation_failed`, including on the final turn. Both stop without
  opening a coder round, preserving the coder's program for scoring, as the
  staged pipeline does when validation retries run out.

When a response hits the output limit, both agents are retried with a reminder, as the native
generator is: the coder gets native's, and the auditor is asked for its report and verdict.
These model-call retries do not consume extra agent turns.

Settings live in `zeroshot/configs/native2.yaml`, which extends `native.yaml`.
Logging, `on_existing`, the retrospective (coder only), and scoring follow the
native README. `run.json` says `"pipeline": "native2"`. `events.jsonl` holds
both agents, split by `prompt` records naming the `agent`. It also holds `audit_verdict` and
`run_completed.coder_rounds`. `stage_validation_retry` records the role, round,
retry number and cumulative turns; recovered retries do not count as budget
exhaustion in aggregation. The auditor's transcript is `audit_messages.json`.

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
