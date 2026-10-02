# Native drawing-to-CadQuery baseline

A single LangGraph agent receives the PNG drawing and writes `/work/model.py`.
The prompt requires the final shape to be assigned to `result` and forbids STEP
export code in `model.py`.
Only `run_shell` and `load_image` are exposed. The agent chooses its own workflow;
there is no auditor, verifier, turn budget, structured submission, compaction,
or answer-correction retry. An ordinary final answer ends generation. A missing,
empty, or symlinked `model.py` is recorded as failure without asking again.

The existing sandbox, two tools, OpenRouter and direct Claude adapters, and stateless-reasoning
middleware are reused unchanged. The native runner and logging remain separate.
The native agent installs the stateless guard, tool-error handling, and `ConnectionRetryMiddleware`
to retry transport failures (dropped streams, timeouts, HTTP 429/5xx) with backoff (`model_retries`, default 5).
No new dependencies are required.

After generation is saved, the runner asks the same model once for a chronological
retrospective explanation of its approach and changes in its thinking. It uses
numbered Markdown headings; the model chooses their number and titles. The request
is never shown during generation and prescribes no stages or categories. This call receives the completed in-memory
conversation and the final script, has no tools, and its visible response is
saved by the runner to `reasoning_traj.md`. It cannot change the generated CAD.
The explanation is a retrospective account, not a recovered internal trace.

## Run

From the repository/worktree root with the existing `drawing2cad` environment:

```bash
python -m zeroshot.pipeline_native \
  model=gpt6_luna_codex \
  sample.sample_id=000364 \
  artifact_root=outputs/native_luna_6.0
```

The default is GPT-6 Luna, `reasoning.effort=max`, `summary=detailed`, using the
same Codex model adapter/authentication as the existing GPT experiments. The
model accepts images and reasoning with tool calls through Responses
([model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)).

Prepared model configs (select with `model=...`):

| Config | Model | Reasoning effort | Integration / Authentication |
| --- | --- | --- | --- |
| `gpt6_luna_codex` | GPT-6 Luna (6.0) | `max` | Codex connection |
| `gpt6_sol_codex` | GPT-6 Sol (6.0) | `max` | Codex connection |
| `gpt6.1_sol_codex` | GPT-6.1 Sol (6.1) | `max` | Codex connection |
| `gpt6_astra_codex` | GPT-6 Astra | `max` | Codex connection |
| `opus5.5_claude` | Claude Opus 5.5 | `high` (adaptive) | Direct Claude (`zeroshot.pipeline.models.claude`) |
| `opus5.5_openrouter` | Claude Opus 5.5 | `max` | OpenRouter (`OPENROUTER_API_KEY`) |
| `sonnet5.5_claude` | Claude Sonnet 5.5 | `high` (adaptive) | Direct Claude (`zeroshot.pipeline.models.claude`) |
| `sonnet5_openrouter` | Claude Sonnet 5 | `max` | OpenRouter (`OPENROUTER_API_KEY`) |

The GPT configurations use the Codex connection. Claude direct configurations use
OAuth / Anthropic API credentials (`claude auth login`). Claude OpenRouter configurations use
OpenRouter and `OPENROUTER_API_KEY`; they request reasoning output and keep
128,000 output tokens available for thinking plus the response. `model_retries`
(default: `5`) controls the number of transport retry attempts on dropped streams or rate limits.

Maximum efforts were checked against the official [Luna](https://developers.openai.com/api/docs/models/gpt-6-luna),
[Sol](https://developers.openai.com/api/docs/models/gpt-6-sol),
[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra), and
[Claude effort](https://platform.claude.com/docs/en/build-with-claude/effort) specifications.
OpenRouter documents [summarized thinking by default](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#summarized-thinking)
for newer Claude models; `reasoning.exclude=false` keeps that trace in responses.

For GLM, use the existing standalone `model=glm5.3_flash_openrouter` config and
`OPENROUTER_API_KEY`. Both GLM and Claude use
`zeroshot.pipeline.models.openrouter.ChatOpenRouterSingleReasoning` directly.
Every model config is independently defined; none inherits another model config.
Do not pass a secret on the command line.

All native run settings live in `zeroshot/configs/workflow/native.yaml`,
alongside the existing `workflow/single.yaml`. The native entrypoint loads it
with `config_name="workflow/native"` from `zeroshot/configs/`, using the shared
model/input groups. The commands above use this config by default; select
alternatives with `model=...` or `input=...`. Native uses only the input sheets
and model fields; shared
presenter/response-format settings do not add behavior. Hydra `--multirun`
works with the usual overrides.

To run or reissue a sweep:

```bash
python -m zeroshot.pipeline_native --multirun \
  model=gpt6_luna_codex \
  artifact_root=outputs/native_luna_6.0 \
  on_existing=retry \
  sample.sample_id=$(ls data/test_vlm/target_step | sed 's/\.step//' | paste -sd,)
```

`on_existing` follows the staged pipeline's policies:

- `fail` (default): refuse completed or incomplete existing runs.
- `skip`: skip generations with `run_completed`; refuse failed or interrupted runs.
- `retry`: skip generations with `run_completed`; clear failed or interrupted runs
  and generate from scratch, keeping Hydra's `.hydra/` and job logs.

Completion is determined by generation's `events.jsonl`. A retrospective failure
after `run_completed` still counts as completed and is skipped by `skip`/`retry`.
Use a new `artifact_root` to regenerate a completed sample.

A worktree does not contain the ignored dataset. Point directly to the source
PNG; the runner copies only this file into the sandbox:

```bash
python -m zeroshot.pipeline_native \
  sample.drawing.sheets.0.file=/absolute/path/000364_dim.png \
  artifact_root=/absolute/path/native_run \
  console=false
```

The sandbox Python defaults to the Python launching the runner. Override
`sandbox_runner.python_executable` if CadQuery is in a different environment.
Shell/API timeouts remain configured; LangGraph's required integer recursion
limit is set to `sys.maxsize`, with no agent turn-budget policy.

## Records

Each `<artifact_root>/<sample_id>/` contains:

- `workspace/inputs/`: PNG inputs, mounted read-only. The host repository,
  target STEP, other runs, credentials, and external network are not exposed.
- `workspace/model.py` and any files the agent creates.
- `events.jsonl`: append-and-flush log, including partial model streams, tools,
  prompts, and `run_started` / `run_completed` / `run_failed` events.
- `messages.json`: full final transcript, also written when the agent finishes
  without submitting `model.py`. A stream failure may have only `events.jsonl`.
- `run.json`: redacted resolved settings, input hashes, dependency versions,
  and Git commit/dirty status. Hydra also writes its normal `.hydra/` files.
- `reasoning_traj.md`: the model's chronological retrospective explanation, outside
  the agent workspace.
- `retrospective_events.jsonl`: separate streamed events, request, and
  `retrospective_started` / `retrospective_completed` / `retrospective_failed`.
- `retrospective_messages.json`: the new request and final response; the preceding
  conversation is already in `messages.json`.

`events.jsonl` ends with `run_completed` once generation is saved. A later
retrospective failure raises an error but does not relabel generation as failed
or overwrite its transcript. Check `retrospective_completed` and the Markdown
file for report success. Analyze report usage/time separately from generation;
each stream has its own protocol sequence. Partial report streams survive in
`retrospective_events.jsonl` even if no finished Markdown file is produced.

The native log has its own schema; it is not the staged pipeline's event schema.
Each record has `schema_version`, `event_index`, `timestamp`, `event`, and `data`.
Protocol records retain `seq` and `namespace`. For `event="messages"`, `data` is
`[payload, metadata]`; the payload is either a streaming protocol dictionary
(`payload.event`) or a complete serialized AI message (`payload.type="ai"`).
Provider fields including `additional_kwargs` and `response_metadata` are kept.
For `event="tools"`, `data` contains the original tool event.

Use event order, namespace, and `metadata.langgraph_step` to relate model calls;
provider/run IDs are retained when supplied. Use finished reasoning blocks or
the final AI message for successful calls. For interrupted calls, concatenate
reasoning deltas in order and label them partial. Do not add deltas, finished
blocks, and final transcript copies together. A call with no returned summary
is distinct from a partial call. Only provider-exposed reasoning is available.
Images in logs are replaced by hashes/lengths and known secret fields are
redacted. `console=false` disables printing, not logging.

Generation success means a nonempty `model.py` was submitted, not that its
geometry is correct. For an execution smoke check, run the generated file in
the shared sandbox after the agent has stopped, and export `result` from a
separate checking script. There is no automated geometry feedback. The native
runner does not touch the target STEP or invoke the staged scorer. Any execution
check artifacts (such as `smoke/` or `smoke_result.json`) come from separate
post-run checks; the native runner does not create them.

## Checks

```bash
python -m pytest -q tests/zeroshot/pipeline_native \
  tests/zeroshot/test_sandbox.py \
  tests/zeroshot/tools/test_run_shell.py \
  tests/zeroshot/tools/test_load_image.py
```
